"""Fresh, quality-gated circle measurements. No arm/driver dependencies."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import math
import cv2
import numpy as np
from localization.vision_contract import validate_grid, parse_coordinate


@dataclass(frozen=True)
class VisionCalibration:
    width_mm: float = 276.0
    height_mm: float = 307.0
    piece_diameter_mm: float = 25.0
    robot_from_board: tuple = ((-1.,0.),(0.,1.))
    calibrated: bool = False
    calibration_id: str = "unverified-config-dimensions"
    columns_mm: tuple = ()
    rows_mm: tuple = ()
    max_offset_mm: float = 8.0
    camera_matrix: tuple = ()
    distortion: tuple = ()
    image_size: tuple = ()
    camera_height_mm: float = None
    piece_height_mm: float = None
    camera_center_board_mm: tuple = ()

    def __post_init__(self):
        if type(self.calibrated) is not bool:raise ValueError('calibrated must be a JSON boolean')
        numbers=[self.width_mm,self.height_mm,self.piece_diameter_mm,self.max_offset_mm]
        if not np.isfinite(numbers).all() or min(numbers)<=0:raise ValueError('Calibration dimensions must be finite and positive')
        transform=np.asarray(self.robot_from_board,float)
        if transform.shape!=(2,2) or not np.isfinite(transform).all() or abs(np.linalg.det(transform))<1e-6:
            raise ValueError('Invalid board-to-robot displacement transform')
        object.__setattr__(self,'robot_from_board',tuple(tuple(float(v) for v in row) for row in transform))
        object.__setattr__(self,'columns_mm',tuple(self.columns_mm))
        object.__setattr__(self,'rows_mm',tuple(self.rows_mm))
        for values,n,span in ((self.columns_mm,9,self.width_mm),(self.rows_mm,10,self.height_mm)):
            if values and (len(values)!=n or not np.isfinite(values).all() or np.any(np.diff(values)<=0)
                           or abs(values[0])>1e-6 or abs(values[-1]-span)>1e-3):
                raise ValueError('Grid distances must increase from zero to full span')
        if self.calibrated and not self.calibration_id.strip():raise ValueError('Verified calibration needs an identifier')
        if any(len(value)>0 for value in (self.camera_matrix,self.distortion,self.image_size)):
            matrix=np.asarray(self.camera_matrix,float)
            if (matrix.shape!=(3,3) or not np.isfinite(matrix).all() or matrix[0,0]<=0 or matrix[1,1]<=0
                    or len(self.distortion) not in (4,5,8) or not np.isfinite(self.distortion).all()
                    or len(self.image_size)!=2 or min(self.image_size)<=0):
                raise ValueError('Lens calibration requires K, distortion and matching image size')
            object.__setattr__(self,'camera_matrix',tuple(map(tuple,matrix)))
            object.__setattr__(self,'distortion',tuple(self.distortion))
            object.__setattr__(self,'image_size',tuple(self.image_size))
        if self.camera_height_mm is not None or self.piece_height_mm is not None or self.camera_center_board_mm:
            values=[self.camera_height_mm,self.piece_height_mm,*self.camera_center_board_mm]
            if (None in values or len(self.camera_center_board_mm)!=2 or not np.isfinite(values).all()
                    or not 0<=self.piece_height_mm<self.camera_height_mm):
                raise ValueError('Height correction requires camera height, piece height and camera center in board coordinates')
            object.__setattr__(self,'camera_center_board_mm',tuple(self.camera_center_board_mm))

    @classmethod
    def from_config(cls):
        import config
        return cls(width_mm=config.BOARD_WIDTH_MM,height_mm=config.BOARD_HEIGHT_MM,
                   piece_diameter_mm=config.PIECE_DIAMETER_MM)

    @classmethod
    def load(cls,path):
        raw=json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(**raw)

    def nominal(self,r,c):
        x=self.columns_mm[c] if self.columns_mm else c*self.width_mm/8
        y=self.rows_mm[r] if self.rows_mm else r*self.height_mm/9
        return np.array([50+350*x/self.width_mm,50+400*y/self.height_mm])

    def displacement(self,delta):
        board=np.asarray(delta,float)*[self.width_mm/350,self.height_mm/400]
        return board,np.asarray(self.robot_from_board)@board

    def surface_nominal(self,r,c):
        nominal=self.nominal(r,c)
        if self.camera_height_mm is None:return nominal
        board=(nominal-[50,50])*[self.width_mm/350,self.height_mm/400]
        center=np.asarray(self.camera_center_board_mm)
        scale=1-self.piece_height_mm/self.camera_height_mm
        projected=center+(board-center)/scale
        return projected*[350/self.width_mm,400/self.height_mm]+[50,50]

    def offset_from_detection(self,nominal,detected):
        scale=np.array([self.width_mm/350,self.height_mm/400])
        expected=(np.asarray(nominal)-[50,50])*scale
        actual=(np.asarray(detected)-[50,50])*scale
        if self.camera_height_mm is not None:
            center=np.asarray(self.camera_center_board_mm)
            actual=center+(1-self.piece_height_mm/self.camera_height_mm)*(actual-center)
        board=actual-expected
        return board,np.asarray(self.robot_from_board)@board

    def compatible_with_robot(self):
        """Read nominal mapping only; never change its measured parameters."""
        import config
        xs=np.arange(9)*config.SQUARE_WIDTH
        ys=np.array([r*config.SQUARE_HEIGHT if r<=4 else
                     4*config.SQUARE_HEIGHT+config.RIVER_HEIGHT+(r-5)*config.SQUARE_HEIGHT for r in range(10)])
        actual_x=np.asarray(self.columns_mm or tuple(np.linspace(0,self.width_mm,9)))
        actual_y=np.asarray(self.rows_mm or tuple(np.linspace(0,self.height_mm,10)))
        return (np.max(abs(xs-actual_x))<.1 and np.max(abs(ys-actual_y))<.1
                and np.allclose(self.robot_from_board,((-1,0),(0,1)),atol=1e-6))


class OffsetUnavailable(RuntimeError):
    pass


def require_offset(info, now, max_age=1.0, calibrated=True):
    if not info or not info.get('valid'):
        raise OffsetUnavailable((info or {}).get('reason','No valid center measurement'))
    captured=info.get('captured_at')
    if captured is None or not np.isfinite(captured):raise OffsetUnavailable('Measurement lacks a valid acquisition timestamp')
    age=now-captured
    if not np.isfinite(age) or age<0 or age>max_age:raise OffsetUnavailable('Center measurement expired')
    if calibrated and not info.get('control_valid'):raise OffsetUnavailable('Physical calibration not verified')
    value=info.get('robot_offset_mm')
    if value is None or len(value)!=2 or not np.isfinite(value).all():raise OffsetUnavailable('Invalid robot offset')
    return tuple(float(v) for v in value)


class CenterMeasurer:
    def __init__(self,calibration=None):
        self.calibration=calibration or VisionCalibration.from_config()

    def measure(self,image,grid,*,frame_id=-1,captured_at=None,positions=None):
        grid=validate_grid(grid)
        if image.shape!=(500,450,3) or image.dtype!=np.uint8:raise ValueError('Need uint8 canonical board 500x450')
        gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(5,5),.8)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
        mag=cv2.magnitude(gx,gy)
        cal=self.calibration
        radius=cal.piece_diameter_mm*.25*(350/cal.width_mm+400/cal.height_mm)
        if cal.camera_height_mm is not None:radius/=1-cal.piece_height_mm/cal.camera_height_mm
        if not 8<=radius<=23:raise ValueError('Piece diameter inconsistent with canonical grid')
        radii=np.linspace(radius*.85,radius*1.15,5)
        contours,_=cv2.findContours(cv2.Canny(gray,40,100),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        closed_rims=[]
        for contour in contours:
            if len(contour)<24:continue
            center,axes,_=cv2.fitEllipse(contour)
            rad=sum(axes)/4
            circularity=4*np.pi*cv2.contourArea(contour)/max(cv2.arcLength(contour,True)**2,1)
            if (abs(rad-radius)<=radius*.20 and min(axes)>0 and max(axes)/min(axes)<1.25
                    and circularity>.70):
                closed_rims.append((np.array(center),rad,circularity))
        responses=[]
        for rad in radii:
            size=int(np.ceil(rad+3))*2+1
            y,x=np.mgrid[:size,:size]-(size//2)
            kernel=(abs(np.hypot(x,y)-rad)<=1.2).astype(np.float32)
            kernel/=kernel.sum()
            responses.append(cv2.filter2D(mag,-1,kernel,borderType=cv2.BORDER_CONSTANT))
        if positions is None:positions=[f'{chr(97+c)}{r}' for r in range(10) for c in range(9) if grid[r][c]!='.']
        output={}
        angles=np.arange(96)*2*np.pi/96
        unit=np.stack([np.cos(angles),np.sin(angles)],axis=1)
        for pos in positions:
            c,r=parse_coordinate(pos);pos=pos.lower()
            nominal=cal.nominal(r,c)
            search_center=cal.surface_nominal(r,c)
            info=dict(pos_str=pos,cell=(r,c),char=grid[r][c],nominal_px=nominal.tolist(),
                      frame_id=frame_id,captured_at=captured_at,valid=False,control_valid=False,
                      reason='No reliable circular rim',robot_offset_mm=None,board_offset_mm=None,
                      calibration_id=cal.calibration_id)
            info.update(surface_nominal_px=search_center.tolist(),parallax_corrected=cal.camera_height_mm is not None)
            output[pos]=info
            if grid[r][c]=='.':info['reason']='Empty intersection';continue
            search=min(13.,cal.max_offset_mm*min(350/cal.width_mm,400/cal.height_mm))
            if not (radius+3<search_center[0]<450-radius-3 and radius+3<search_center[1]<500-radius-3):
                info['reason']='Projected piece rim outside calibrated image';continue
            x0,x1=max(0,int(search_center[0]-search)),min(450,int(search_center[0]+search+1))
            y0,y1=max(0,int(search_center[1]-search)),min(500,int(search_center[1]+search+1))
            candidates=[]
            for rad,response in zip(radii,responses):
                _,score,_,peak=cv2.minMaxLoc(response[y0:y1,x0:x1])
                candidates.append((score,np.array([x0+peak[0],y0+peak[1]],float),rad))
            _,center,rad=max(candidates,key=lambda v:v[0])
            nearby=[entry for entry in closed_rims if np.linalg.norm(entry[0]-search_center)<=search]
            if nearby:
                center,rad,_=max(nearby,key=lambda v:v[2]-.005*np.linalg.norm(v[0]-search_center))
            points=None
            for _ in range(3):
                offsets=np.linspace(-2.5,2.5,11)
                samples=center+unit[:,None,:]*(rad+offsets)[None,:,None]
                mx,my=samples[:,:,0].astype(np.float32),samples[:,:,1].astype(np.float32)
                sx=cv2.remap(gx,mx,my,cv2.INTER_LINEAR)
                sy=cv2.remap(gy,mx,my,cv2.INTER_LINEAR)
                radial=abs(sx*unit[:,0,None]+sy*unit[:,1,None])
                winner=radial.argmax(axis=1)
                best=radial[np.arange(96),winner]
                selected=samples[np.arange(96),winner]
                transverse=np.hypot(sx[np.arange(96),winner],sy[np.arange(96),winner])
                good=(best>=20)&(best/np.maximum(transverse,1)>=.7)
                if good.sum()<60:break
                points=selected[good]
                # Algebraic circle fit followed by robust radial rejection.
                for _ in range(3):
                    coefficients=np.linalg.lstsq(np.c_[2*points,np.ones(len(points))],(points*points).sum(axis=1),rcond=None)[0]
                    fitted=coefficients[:2]
                    radius2=coefficients[2]+np.dot(fitted,fitted)
                    if radius2<=0:points=None;break
                    fitted_r=np.sqrt(radius2)
                    residual=abs(np.linalg.norm(points-fitted,axis=1)-fitted_r)
                    keep=residual<1.4
                    if keep.sum()<60:points=None;break
                    if keep.all():break
                    points=points[keep]
                if points is None:break
                center,rad=fitted,fitted_r
            coverage=float(good.mean())
            quadrants=[float(good[n:n+24].mean()) for n in range(0,96,24)]
            if points is None or coverage<.78 or min(quadrants)<.55:continue
            residual=float(np.sqrt(np.mean((np.linalg.norm(points-center,axis=1)-rad)**2)))
            delta=center-nominal
            board,robot=cal.offset_from_detection(nominal,center)
            if not np.isfinite([*center,rad,residual]).all() or residual>1.0 or abs(rad-radius)>radius*.20:continue
            if np.linalg.norm(center-search_center)>search-1 or np.linalg.norm(board)>cal.max_offset_mm:
                info['reason']='Center offset outside validated search range';continue
            quality=coverage*max(0,1-residual/2)
            control_valid=bool(cal.calibrated and cal.compatible_with_robot() and captured_at is not None)
            reason=('Pixel measurement valid; calibration unverified' if not cal.calibrated else
                    'Calibration differs from robot nominal grid' if not cal.compatible_with_robot() else
                    'No live acquisition timestamp' if captured_at is None else '')
            info.update(valid=True,control_valid=control_valid,reason=reason,
                        detected_px=center.tolist(),delta_px=delta.tolist(),radius_px=float(rad),
                        board_offset_mm=board.tolist(),robot_offset_mm=robot.tolist(),
                        offset_dist_mm=float(np.linalg.norm(board)),confidence=float(quality),
                        rim_coverage=coverage,fit_residual_px=residual)
        return output

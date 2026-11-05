"""Readable shading must pass; missing or hidden piece content must not."""
import unittest
import cv2
import numpy as np
from localization.appearance_verifier import BoardAppearance
from localization.live_vision import MotionCheck, HandObservation, board_hand_count
from test_appearance_verifier import image


def shadow(board, kind):
    y,x = np.mgrid[:500,:450]
    fields = {
        'global': np.full((500,450),.78),
        'side': .70+.30/(1+np.exp(-(x-220)/20)),
        'local': 1-.28*np.exp(-((x-110)**2+(y-320)**2)/(2*45**2)),
        'moving_edge': .82+.18/(1+np.exp(-(x+y-440)/8)),
    }
    return np.clip(board.astype(float)*fields[kind][...,None],0,255).astype(np.uint8)


class TestShadowVerification(unittest.TestCase):
    def test_readable_global_local_and_side_shadows_remain_valid(self):
        board=image();v=BoardAppearance(board)
        for kind in ('global','side','local','moving_edge'):
            with self.subTest(kind=kind):self.assertTrue(v.match(shadow(board,kind))[0])

    def test_shading_cannot_hide_removed_changed_or_obscured_piece(self):
        board=image();v=BoardAppearance(board)
        for kind in ('remove','obscure','side'):
            current=board.copy()
            if kind=='remove':current[335:389,24:78]=(150,175,190)
            elif kind=='obscure':cv2.rectangle(current,(40,350),(62,375),(50,80,130),-1)
            else:
                current=image(red=False)
            for lighting in ('side','local'):
                with self.subTest(change=kind,light=lighting):
                    self.assertFalse(v.match(shadow(current,lighting))[0])

    def test_unreadably_dark_or_saturated_board_is_not_lighting(self):
        v=BoardAppearance(image())
        for level in (0,255):self.assertFalse(v.match(np.full((500,450,3),level,np.uint8))[0])

    def test_shadow_does_not_modify_immutable_reference(self):
        board=image();v=BoardAppearance(board)
        for kind in ('side','local','global'):v.match(shadow(board,kind))
        np.testing.assert_array_equal(v.reference,board)


class TestScopedMotion(unittest.TestCase):
    def test_soft_local_shadow_does_not_trigger_motion(self):
        board=image();motion=MotionCheck();motion.update(board,10)
        self.assertEqual(motion.update(shadow(board,'local'),10.1),'')

    def test_background_changes_outside_board_are_ignored(self):
        rng=np.random.default_rng(19)
        raw=rng.integers(50,210,(400,600,3),dtype=np.uint8)
        region=np.float32([[160,60],[440,60],[440,340],[160,340]])
        motion=MotionCheck();motion.update(raw,10,region)
        changed=raw.copy();changed[:,:140]=(30,80,150)
        self.assertEqual(motion.update(changed,10.1,region),'')

    def test_occluder_inside_board_still_triggers_motion(self):
        rng=np.random.default_rng(19);raw=rng.integers(50,210,(400,600,3),dtype=np.uint8)
        region=np.float32([[160,60],[440,60],[440,340],[160,340]])
        motion=MotionCheck();motion.update(raw,10,region)
        raw[130:270,210:380]=(35,75,150)
        self.assertTrue(motion.update(raw,10.1,region))


class TestBoardHands(unittest.TestCase):
    polygon=np.float32([[100,100],[400,100],[400,400],[100,400]])

    def hand(self,points):
        return HandObservation(1,10,10.01,1,10,(tuple(points),))

    def test_side_hand_does_not_pause_board(self):
        hand=self.hand(((.85,.2),(.95,.2),(.95,.5),(.85,.5)))
        self.assertEqual(board_hand_count(hand,self.polygon,(500,600,3)),0)

    def test_partial_entry_still_pauses(self):
        hand=self.hand(((.63,.3),(.8,.3),(.8,.6),(.63,.6)))
        self.assertEqual(board_hand_count(hand,self.polygon,(500,600,3)),1)

    def test_missing_geometry_or_landmarks_remains_conservative(self):
        hand=self.hand(((.85,.2),(.95,.2),(.95,.5),(.85,.5)))
        self.assertEqual(board_hand_count(hand,None,(500,600,3)),1)
        self.assertEqual(board_hand_count(HandObservation(1,10,10,1,0),self.polygon,(500,600,3)),1)

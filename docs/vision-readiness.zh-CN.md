# 第四代视觉：离线验收与首次上机操作

## 当前边界

本轮完善视觉软件、坐标协议、游戏同步、圆心测量和模拟闭环。未连接真实摄像头、CH347 或机械臂，没有修改运动学、舵机标定、电磁铁时序、抓放轨迹或 ONNX 权重。

软件检查通过后，可以开始受控的真实视觉验证。实际抓放还需要现场完成方向、尺寸、镜头、棋子高度及机械臂坐标标定，不能把离线成功解释成真实抓取成功或 1 mm 测量精度。

唯一默认识别方案保持 RTMPose + 原始 SwinV2 Nano + 保守后处理，MobileNetV3 仅用于有限的辅助判断。Tiny 微调版和 15 类 patch 模型不作为默认模型。此前反光条件下仍有错误；本轮按照用户要求冻结分类方案，没有宣称任意光照 100%。

## 功能和验收层次

| 目标 | 当前软件行为 | 仍需实物确认 |
|---|---|---|
| 自动定位 | 自动四角、完整性/凸性/退化检查、透视校正；不需要点击角点 | 实际安装方向、镜像、裁边、镜头畸变 |
| 全盘识别 | 10×9、14 种棋子、空点；原始遮挡/未知保持拒绝；不要求 32 子 | 独立中残局及室内/自然光准确率 |
| 稳态 | 不同物理帧确认、手部独立检查、新帧外观复核、阴影/小抖动容忍 | 静止手、机械臂或其他物体遮挡的漏检率 |
| 走棋与吃子 | 比较整盘，唯一变更提议，经规则及完整软件盘面确认后提交 | 连续真实移子与吃子事件 |
| 任意局面 | 视觉可以观察任意规则允许数量的摆放；从局面继续对弈需显式初始化和指定轮次 | 无法单靠照片知道轮到谁；现场核对布局 |
| 毫米偏置 | 圆心拟合、残差/覆盖率门限、方向/尺度转换、每子质量及时间戳 | 棋盘尺寸、真实圆心误差、视差、机械误差 |
| 动作后复核 | 动作令牌、帧号和采集时间双重屏障；全盘和目标新测圆心匹配后才提交 | 控制器动作结束是否代表遮挡退出、真实落子精度 |
| 软件运行 | 图片/视频/相机诊断、观察模式、GUI、模拟入口、异常拒绝和退出 | 当前设备实际帧率、长时间稳定性 |

绿色稳态意味着通过当前软件证据检查，不意味着整盘正确概率为 100%。新帧外观跟踪通过也不等于每张预览帧重新运行了神经网络。屏幕历史标签不参与走法提交或偏置控制。

## 环境与模型

在项目根目录运行下面的 PowerShell 命令。使用已安装依赖的同一个 Python 环境；需要补装时：

```powershell
python -m pip install -r requirements-vision-live.txt
```

必需权重位于 `recognition/models/onnx/`：`pose_rtmpose_4kpt.onnx`、`layout_swinv2_nano.onnx`；当前默认辅助权重为 `piece_classifier_mobilenetv3.onnx`。ONNX 文件被 Git 忽略，换电脑需要单独复制。不会因为缺模型而返回假分类或自动改用 Tiny。

当前测试环境为 Python 3.11.9、OpenCV 4.13.0.92、ONNX Runtime 1.31.1、NumPy 2.4.4、MediaPipe 0.10.35、PyQt5 5.15.11。Windows 下保留现有推理库先于 Qt 的导入顺序，避免已复现的本机 DLL 冲突。

手部模型使用 MediaPipe `hand_landmarker.task`，相机入口必须明确提供其路径。已经存在的模型可以直接复用：

```powershell
$visionHandModel = 'C:\Users\Liu\.codex\cache\tasks\01a16c6a-vision-audit-20261104\live_camera_check\hand_landmarker.task'
$visionRun = Join-Path $env:USERPROFILE ('.codex\cache\tasks\chessrobot-vision-' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
```

不要同时打开两个占用同一摄像头的入口。相机编号 2 是此前本机 AX USB2.0 的编号，重新连接后仍需确认；1920×1080 是请求值，窗口显示的实际分辨率才是设备返回值。

## 今天无需连接设备的验证

### 全量验收

```powershell
python -B tools/verify_vision_offline.py --output-dir (Join-Path $visionRun 'acceptance')
```

输出 `acceptance.json` 和 `tests.log`。检查 `passed`、测试计数、模型哈希、失败/跳过原因和 `forbidden_device_attempts`。验收器禁止打开数字编号相机和加载 CH347；视频文件读取允许。模型零输入检查只检查形状、可加载性和有限数值，不能作为准确率测试。

### 真实软件接口的模拟闭环

```powershell
python -B main.py --mode simulate
```

使用 Fake Camera / Recognizer / Robot，经过真实 ONNX 适配器、稳态判定、圆心测量、主调度器、游戏规则和动作后新帧复核。输出应包含 `passed: True`、`board_matches_camera: True`、`hardware_accessed: False`。模拟识别器不代表真实模型准确率，模拟机械臂不代表真实精度。

### 已有图片或视频

```powershell
python -B tools/vision_debug.py --image 'recognition/data/user_test_samples/user_move_1.jpg' --headless --output-dir (Join-Path $visionRun 'image')
python -B tools/vision_debug.py --video 'D:\你的录制视频.mp4' --output-dir (Join-Path $visionRun 'video')
```

图片保存原图、450×500 校正图和 JSON；单张图片始终 `stable=false`、`control_ready=false`，不会循环同一照片伪造稳态。视频显示实时诊断，也始终观察使用。`--headless` 不弹窗；`--max-seconds` 可限制视频/相机运行时长。

## 白天第一步：仅验证视觉

优先使用此前调整过的高清观察预览：

```powershell
python -B tools/vision_live.py --camera 2 --width 1920 --height 1080 --hand-model $visionHandModel --output-dir (Join-Path $visionRun 'live')
```

1. 固定相机，完整拍入棋盘四边及最外侧棋子，保留少量边缘。正常室内灯光开始。
2. 不需要点击四角，不需要按空格。稳定后自动确认；原图显示小坐标/引线，中文棋名在侧边棋盘上。
3. 必须用非对称布局检查方向：黑方底线为第 0 行，红方底线为第 9 行；从机械臂黑方视角，a 列在右、i 列在左。逐个核对 a0、i0、a9、i9，不能只用对称标准开局判断。
4. 核对全部 90 格，包括空点、阵营和棋名。先标准开局，再中局、残局、旋转棋子及少量偏心摆放。手动记录真值；有错就保存样本，不能用模型自己的预测当真值。
5. 分别观察正常室内灯光和自然光。测试柔和阴影、轻微明暗变化；再测试真实手伸入并静止、移出、棋盘裁边、断流。前者可辨认时应尽量稳定，后者必须撤销可信输出。
6. 连续观察至少数分钟，保存异常片段及前后状态。棋名字体不闪与当前观测可信是不同指标。
7. `O` 显示新帧圆心及偏置，`S` 保存，`R` 重建独立观察参考，`L` 坐标开关，`F` 全屏，`Q/Esc` 退出。

蓝色十字为理论交叉点，黄色圆心及连线是本帧测量。偏置诊断没有标定时仍可看像素结果，但不能作为机械臂控制输入。记录每个无效测量的 `reason`。

需要直接验证整机所用适配器时，使用统一诊断窗口：

```powershell
python -B tools/vision_debug.py --camera 2 --hand-model $visionHandModel --output-dir (Join-Path $visionRun 'adapter')
```

该入口没有机器人；`R` 只清除当前证据、重新确认，保留已提交的基准，不会把任意新布局直接当作合法走棋。人为重新布阵后，退出重启这个诊断进程以建立新的独立观察局面。多处同时变化会明确进入冲突，不猜测某一步。

查看 `events.jsonl`、`status.json` 和 `monitor_latest.jpg`（高清预览）；统一诊断入口保存 `latest.json`、原图、校正图和预览，走棋候选写入 `events.jsonl`。独立观察窗口显示的是候选，不会提交游戏或触发动作。

## 毫米补偿：现场标定后才可控制

`localization/vision_calibration.example.json` 是结构示例，`calibrated=false`。复制到项目外的本次测试目录，再按实测填写，不能为了让程序运行而直接改成 true。

- `width_mm`、`height_mm`：最外侧交叉点跨度，分别对应 a～i 和 0～9，不是棋盘外框尺寸；棋子直径也要测量。
- `columns_mm` / `rows_mm`：可填逐列/逐行实测位置，从 0 到总跨度，显式处理河界或非均匀间距。若与当前机械臂名义格点不一致，视觉可以显示测量，但自动控制会拒绝，不能靠篡改视觉尺寸掩盖控制坐标差异。
- `robot_from_board`：棋盘位移到机械臂位移的 2×2 变换；当前控制约定为 x 反向、y 同向。实际正负方向需现场验证，不在本轮改动硬件标定。
- `camera_matrix`、`distortion`、`image_size`：可选镜头内参、畸变和对应采集尺寸，必须整套提供；分辨率不符则拒绝。手部轮廓同时转换到去畸变坐标，避免错位。
- `camera_height_mm`、`piece_height_mm`、`camera_center_board_mm`：可选棋子高度视差修正，适用于平面棋盘、近似平行棋子顶面、针孔相机模型；相机中心投影采用棋盘毫米坐标。必须整套实测；缺省不假装视差已消除。
- `calibration_id`：记录本次实物安装/测量标识；只有尺寸、轴向和实际误差核对完成后，才能标记 `calibrated=true`。

带标定的观察命令只需增加 `--calibration 'D:\你的实际标定.json'`。观察模式仍强制 `control_valid=false`，不会因为有标定文件而启动硬件。

用尺或独立标记测量多处棋子真实圆心，向前/后/左/右人为偏移，核对像素方向、mm 比例、误差和拒绝率。还要检查边缘棋子和棋子顶面视差。本次离线圆心测试不能替代这些步骤。

## 主程序、GUI 与后续受控运动

仅观察的主程序和 GUI：

```powershell
python -B main.py --mode observe --camera 2 --hand-model $visionHandModel
python -B ui_main.py --mode observe --camera 2 --hand-model $visionHandModel
```

GUI 点击“开始视觉观察”。默认不构造机械臂控制器；失去可信视觉时不拿标准开局填充观察棋盘。第四代不需要点击角点或空格。CLI 的简易监视窗口用于状态查看；完整小坐标/中文侧栏请用上面的独立视觉预览。

第三代 `--legacy-pt` 回退仍保留其原始手动交互，权重必须本地存在。为避免缺少完整盘面及新帧偏置协议的旧接口直接驱动硬件，旧模式仅保留观察/人工流程；未经新协议适配，不开放自动机械臂动作。没有在本轮实测第三代权重推理。

之后真实运动测试必须由用户现场明确启动。以下命令仅是未来操作说明，本轮未执行：

```powershell
# 先满足上面的视觉、标定、轴向、急停和工作区条件，再考虑运行。
python -B main.py --mode hardware --camera 2 --hand-model $visionHandModel --calibration 'D:\你的已核对标定.json'
# 从合法中残局继续，而非标准开局时，额外明确初始局面及轮次：
python -B main.py --mode hardware --camera 2 --hand-model $visionHandModel --calibration 'D:\你的已核对标定.json' --initialize-from-vision --turn red
```

没有 `--initialize-from-vision` 时要求相机盘面与软件开局相符；启用时要求双方将帅、棋子数量、可达位置和轮次规则一致。视觉能显示乱摆的棋子，并不意味着游戏规则允许从该摆放继续。

首次运动只做单步，再看全盘新帧复核与目标偏置；之后单独验证吃子（起点和被吃子都需新测偏置）。成功后才逐步增加回合。没有自动额外纠偏动作。取子使用当前实体中心偏置，落子目标仍是名义交叉点；下一步重新测量，不累计旧偏置。

出现方向不对、棋子误认、手/机械臂遮挡未暂停、相机断流、偏置符号/尺度异常、测量过期、动作返回失败、复核 `STATE_CONFLICT` / `ERROR_EXCEEDED` / `UNCONFIRMED` / `TIMEOUT` 时，停止自动流程并人工核对。不要通过重置基准把失败动作当成功。

## 对外协议与兼容性

可信快照包含 `schema_version=2`、`grid_chars`、`conf_grid`、`frame_id`、单调时钟 `captured_at`、观察 `generation`、`stable`、`source`、FEN、`tensor_14x10x9`、`tensor_channel_order` 和逐棋子偏置。时间戳用于同一进程中的新鲜度判断，不能当作 Unix 时间。

张量通道固定为 `krncbapKRNCBAP`，与现有 `ChessBoard.get_state_tensor()` 一致；旧适配器红方通道在前的输出不再使用。没有改变 AI 网络结构或既有模型输入定义。`confidence` 为最小逐格概率，不是整盘正确率。

每个偏置含本帧 ID/时间、理论点、检测圆心、像素偏移、棋盘/机械臂 mm 偏移、拟合残差、圆周覆盖率、标定 ID、`valid` 和 `control_valid`。无效时偏置为 `None`，禁止解释为零。

人类走棋：提出事件 → 主程序完整盘面及规则检查 → ACK → 同时提交软件盘面和视觉基准；拒绝时两者均保持原状。机器人走棋：新帧测量 → 动作令牌/视觉抑制 → 动作结束帧屏障 → 新帧整盘/圆心复核 → 成功后提交。旧推理、重复帧和动作前缓存不能跨代提交。

可选 Web 演示与主识别流程共用后端、单张图不做时序平滑。本机未安装 Gradio，本轮仅测试 Web 后端，不宣称网页服务器启动通过；核心视觉入口不依赖 Gradio。

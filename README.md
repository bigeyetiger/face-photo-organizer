# face-photo-organizer

**本地离线人脸识别整理工具：给几张参考照片，自动把一个人在几千张照片、几百个视频里的所有出现找出来，按人归档。**

照片和视频全程不离开你的电脑，不上传任何云端。

> 纯 CPU 可跑。实测参考：普通笔记本 CPU，1372 张照片 87 分钟、185 个 4K 视频（17GB）109 分钟，一次全量扫完。

## 功能特性

- **图片识别归档**：检测照片中所有人脸（多人合影逐人识别），与参考照比对，按相似度阈值归档到各人物文件夹
- **视频识别归档**：自动抽帧识别（长视频只解码关键帧，比逐帧快约 7 倍），累计出现 ≥1 秒才算命中，并记录**出场时间点**（如 `出现于 00:25 03:28`）
- **人物档案自动聚类**：给一个参考照片目录即可，同一人多张照片自动聚成一个档案，背景路人自动分开（可人工剔除）
- **去重重组**：单人照/独占视频按人放、多人同框放 `合影`，每张/每个只保留一份；内容完全相同的副本自动剔除
- **人工核对报告**：CSV 明细（相似度、动作、时间点）+ 浏览器缩略图核对页（绿=确定 / 黄=待复核 / 灰=边缘）
- **隐私安全**：模型文件在本机，全程离线推理

## 环境要求

- Python 3.10+（3.14 实测可用）
- ffmpeg（视频抽帧用，需在 PATH 中）
- Windows / macOS / Linux 均可（Windows 中文路径已做兼容处理）

## 安装

```bash
pip install -r requirements.txt
# 首次运行会自动从 insightface 官方源下载模型 buffalo_l（约 275MB）到 ~/.insightface/
```

国内网络建议用镜像装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 使用流程（四步）

```bash
# 1. 建人物档案：把每个人的参考照（每人 1~3 张清晰正脸）放进一个目录
python find_person.py build --ref D:/photos/sample

#    → 自动聚类并生成 refs_preview.html，浏览器打开核对每个人物是谁
#    → 背景路人等多余档案：编辑 refs.json 删除对应条目

# 2. 扫描照片库，命中照片复制到输出目录（按人物分文件夹）
python find_person.py scan --src D:/photos --out D:/结果 --threshold 0.40

# 3. 扫描视频库（命中视频整段复制，并记录出场时间点）
python find_person_video.py scan --src D:/photos --out E:/结果

# 4. 去重重组：单人素材按人放，多人同框放"合影"，每人/每个只留一份
python organize_family.py   --workdir . --out D:/结果合集
python organize_videos.py   --workdir . --out E:/结果
```

## 参数说明

| 命令 | 参数 | 说明 |
|------|------|------|
| `find_person.py build` | `--ref` | 参考照片目录 |
| `find_person.py scan` | `--src` / `--out` | 照片库 / 输出目录 |
| | `--threshold` | 归档阈值，默认 0.40（宽松，宁多选人工删） |
| `find_person_video.py scan` | `--src` / `--out` | 视频库 / 输出目录 |
| | `--min-sec` | 累计出现多少秒才算命中，默认 1.0 |
| `organize_family.py` | `--workdir` / `--out` | 扫描结果目录（含 results.csv）/ 重组输出 |
| `organize_videos.py` | `--workdir` / `--out` | 同上（含 videos_report.csv） |
| `make_preview.py` | — | 从 results.csv 重建浏览器核对页 |

## 调参与经验

- **阈值**：0.40 宁可多选人工删（推荐初次）；0.45–0.50 更准；≥0.55 只留高置信。
- **相似的人会互报**：父女、母子等近亲在低分辨率画面（视频帧、远景照）上容易混淆，低分条目（<0.47）建议人工过一遍核对页。
- **改人物名字**：直接编辑 `refs.json` 里的 `name` 字段后重跑 scan。
- **新增人物**：往参考目录加照片重跑 `build` 会重新聚类；想跳过重扫请关注后续索引功能。
- **慢速外置盘**：识别结果尽量放本地快盘；本工具已做优化——图片去重只在「同大小」文件间比对内容，避免全量读盘。
- **大图提速**：检测前自动把长边缩到 1600，精度基本无损。

## 项目文件说明

```
find_person.py         图片识别（build 建档案 / scan 扫描）
find_person_video.py   视频识别（720p 规格 / 关键帧加速 / 出场时间点）
organize_family.py     图片去重重组
organize_videos.py     视频去重重组
make_preview.py        图片核对页重建
```

## 技术原理

1. **检测**：SCRFD 检出画面里所有人生脸的位置与关键点；
2. **特征**：ArcFace（w600k_r50）把每张脸编码成 512 维向量——同一人的向量彼此相近，不同人相远；
3. **比对**：候选脸与人物档案向量做余弦相似度，超阈值即命中；
4. **视频加速**：`ffmpeg -skip_frame nokey` 只解码关键帧（约每秒 1 帧采样），配合短视频全帧精扫。

## License

代码以 [MIT License](LICENSE) 发布。

注意：依赖的 InsightFace 官方预训练模型（buffalo_l）的使用条款以 [insightface 官方仓库](https://github.com/deepinsight/insightface) 为准，商用请自行确认。

## 致谢

- [InsightFace](https://github.com/deepinsight/insightface) — 人脸检测与识别模型
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 推理引擎

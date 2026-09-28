# 實驗用的模型檔

`ml/experiments/appearance_face_clip.py`、`appearance_face_segment.py` 在本機跑外貌實驗時讀這裡的模型檔。
和推薦 worker 下載的是同一批檔案（`services/recommendation/tools/fetch_models.py`），大小與 sha256 相同。

| 檔案 | 來源（固定版本） | 大小 | 授權 |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo（commit 47534e2）](https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | 232,589 bytes | MIT，全文見 `services/recommendation/third_party/licenses/yunet-MIT.txt` |
| `selfie_multiclass_256x256.tflite` | [MediaPipe image segmenter（float32 v1）](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite) | 16,371,837 bytes | Apache-2.0，全文見 `services/recommendation/third_party/licenses/mediapipe-Apache-2.0.txt` |

sha256：

- `face_detection_yunet_2023mar.onnx`：`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- `selfie_multiclass_256x256.tflite`：`c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`

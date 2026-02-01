import cv2
import numpy as np
import tritonclient.http as tritonhttp

# -----------------------------
# Config
# -----------------------------
TRITON_URL = "localhost:8000"
MODEL_NAME = "yolov8s_detector"
INPUT_NAME = "images"
OUTPUT_NAME = "output0"
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

# -----------------------------
# Preprocess (YOLOv8-style)
# -----------------------------
def preprocess(image_path):
    img = cv2.imread(image_path)
    assert img is not None, "Image not found"

    h0, w0 = img.shape[:2]
    scale = min(IMG_SIZE / w0, IMG_SIZE / h0)
    nw, nh = int(w0 * scale), int(h0 * scale)

    img_resized = cv2.resize(img, (nw, nh))
    img_padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    img_padded[:nh, :nw] = img_resized

    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)

    return img, img_batch, scale

# -----------------------------
# Postprocess (decode + NMS)
# -----------------------------
def postprocess(pred, orig_img, scale):
    pred = pred[0]              # [84, N]
    pred = pred.T               # [N, 84]

    boxes = pred[:, :4]
    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > CONF_THRES
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    if len(boxes) == 0:
        return orig_img

    # cx, cy, w, h → x1,y1,x2,y2
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    boxes_xyxy /= scale

    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        CONF_THRES,
        IOU_THRES
    )

    for i in indices.flatten():
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_ids[i]}:{confidences[i]:.2f}"
        cv2.putText(orig_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return orig_img

# -----------------------------
# Triton Inference
# -----------------------------
def infer(image_path):
    orig_img, input_tensor, scale = preprocess(image_path)

    client = tritonhttp.InferenceServerClient(url=TRITON_URL)

    inputs = [
        tritonhttp.InferInput(
            INPUT_NAME,
            input_tensor.shape,
            "FP32"
        )
    ]
    inputs[0].set_data_from_numpy(input_tensor)

    outputs = [
        tritonhttp.InferRequestedOutput(OUTPUT_NAME)
    ]

    response = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs
    )

    pred = response.as_numpy(OUTPUT_NAME)
    result_img = postprocess(pred, orig_img, scale)

    cv2.imwrite("output.jpg", result_img)
    print("[OK] Saved result to output.jpg")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    infer("test.jpg")

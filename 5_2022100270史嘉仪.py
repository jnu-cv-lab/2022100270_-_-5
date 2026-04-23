import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "test.jpg"
OUTPUT_DIR = "."
DOC_W, DOC_H = 500, 700


def create_test_image(size=600):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    colors = [(220, 80, 80), (80, 80, 220), (80, 180, 80), (180, 80, 180)]
    
    cv2.rectangle(img, (60, 60), (250, 200), colors[0], 3)
    cv2.circle(img, (430, 130), 100, colors[1], 3)
    for y in range(310, 520, 40):
        cv2.line(img, (40, y), (270, y), colors[2], 3)
    cv2.line(img, (310, 400), (570, 400), colors[3], 3)
    cv2.line(img, (440, 280), (440, 530), colors[3], 3)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    texts = [("Rectangle", (60, 50), colors[0]), ("Circle", (360, 50), colors[1]),
             ("Parallel", (40, 300), colors[2]), ("Perp.Lines", (330, 270), colors[3])]
    for text, pos, color in texts:
        cv2.putText(img, text, pos, font, 0.5, color, 1)
    return img


def apply_similarity(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 20, 0.85)
    M[0, 2] += 30
    M[1, 2] += 20
    return cv2.warpAffine(img, M, (w, h), borderValue=(200, 200, 200)), M


def apply_affine(img):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w-1, 0], [0, h-1]])
    dst = np.float32([[50, 80], [w-80, 30], [80, h-50]])
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (w, h), borderValue=(200, 200, 200)), M


def apply_perspective(img):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst = np.float32([[80, 50], [w-50, 0], [w-30, h-60], [30, h-30]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=(200, 200, 200)), M


def visualize_transformations(original, sim, aff, per):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    titles = ["原始图像", "相似变换", "仿射变换", "透视变换"]
    
    for ax, img, title in zip(axes, [original, sim, aff, per], titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/变换对比图.png", dpi=150, bbox_inches='tight')
    plt.close()


_clicked_pts = []

def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicked_pts) < 4:
        _clicked_pts.append((x, y))
        cv2.circle(param, (x, y), 6, (0, 0, 255), -1)
        cv2.imshow("点击文档四角", param)


def correct_perspective(img):
    global _clicked_pts
    _clicked_pts = []
    display = img.copy()
    
    cv2.namedWindow("点击文档四角", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("点击文档四角", 900, 700)
    cv2.setMouseCallback("点击文档四角", _mouse_callback, display)
    cv2.imshow("点击文档四角", display)
    print("请点击文档四角（左上->右上->右下->左下）...")
    
    while len(_clicked_pts) < 4:
        cv2.waitKey(50)
    cv2.waitKey(800)
    cv2.destroyWindow("点击文档四角")
    
    src = np.float32(_clicked_pts)
    dst = np.float32([[0, 0], [DOC_W-1, 0], [DOC_W-1, DOC_H-1], [0, DOC_H-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (DOC_W, DOC_H), borderValue=(255, 255, 255)), M


def visualize_correction(original_doc, corrected):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle("透视畸变校正")
    
    for ax, img, title in zip(axes, [original_doc, corrected], ["原始图像", "校正结果"]):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/校正对比图.png", dpi=150, bbox_inches='tight')
    plt.close()


def verify_properties(sim_M, aff_M, per_M):
    print("\n" + "=" * 55)
    print("几何性质数值验证")
    print("=" * 55)
    
    def transform(M, pt, is_persp=False):
        r = M @ np.array([pt[0], pt[1], 1.0])
        return (r[0]/r[2], r[1]/r[2]) if is_persp else (r[0], r[1])
    
    def slope(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return float('inf') if abs(dx) < 1e-9 else dy / dx
    
    pts = {
        'line1': ((60, 60), (250, 60)),
        'line2': ((60, 200), (250, 200)),
        'horiz': ((310, 400), (570, 400)),
        'vert': ((440, 280), (440, 530))
    }
    
    transforms = [("相似变换", sim_M, False), ("仿射变换", aff_M, False), ("透视变换", per_M, True)]
    
    for name, M, is_persp in transforms:
        s1 = slope(transform(M, pts['line1'][0], is_persp), transform(M, pts['line1'][1], is_persp))
        s2 = slope(transform(M, pts['line2'][0], is_persp), transform(M, pts['line2'][1], is_persp))
        
        diff = abs(s1 - s2) if s1 != float('inf') and s2 != float('inf') else 0.0
        print(f"\n【{name}】")
        print(f"  平行线斜率差: {diff:.6f} -> {'保持' if diff < 1e-3 else '失去'}")


def main():
    print("[1/3] 生成测试图像并施加三种变换...")
    original = create_test_image(600)
    sim_img, sim_M = apply_similarity(original)
    aff_img, aff_M = apply_affine(original)
    per_img, per_M = apply_perspective(original)
    visualize_transformations(original, sim_img, aff_img, per_img)
    
    print("[2/3] 透视畸变校正...")
    original_doc = cv2.imread(IMAGE_PATH)
    if original_doc is None:
        print(f"未找到 '{IMAGE_PATH}'，跳过校正")
    else:
        corrected, _ = correct_perspective(original_doc)
        visualize_correction(original_doc, corrected)
        cv2.imwrite(f"{OUTPUT_DIR}/校正后文档.png", corrected)
    
    print("[3/3] 数值验证几何性质...")
    verify_properties(sim_M, aff_M, per_M)
    
    # 保存图像
    for name, img in [("01_original", original), ("02_similarity", sim_img),
                      ("03_affine", aff_img), ("04_perspective", per_img)]:
        cv2.imwrite(f"{OUTPUT_DIR}/{name}.png", img)

    print("\n完成，所有图像已保存。")


if __name__ == "__main__":
    main()
import sys
import zipfile
# python_version = sys.version_info
# if python_version.minor == 8:
#     zip_ = zipfile.ZipFile('20240827_opencv_py38.zip')
#     sys.path.insert(0, "20240827_opencv_py38")
# elif python_version.minor == 11:
#     zip_ = zipfile.ZipFile('20240827_opencv_py311.zip')
#     sys.path.insert(0, "20240827_opencv_py311")
# else:
#     raise ValueError("Python version not supported")
# zip_.extractall()

import base64
import hashlib
import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import argparse
from pathlib import Path

  

class SimpleImageQuality:
    def __init__(self):
        self.bucket_info = {
            '0.44': (1, 672, 1536),
            '0.45': (2, 672, 1504),
            '0.48': (3, 704, 1472),
            '0.49': (4, 704, 1440),
            '0.52': (5, 736, 1408),  # 1.91:1
            '0.53': (6, 736, 1376),
            '0.57': (7, 768, 1344),  # 9:16
            '0.59': (8, 768, 1312),
            '0.62': (9, 800, 1280),
            '0.67': (10, 832, 1248),  # 2:3
            '0.68': (11, 832, 1216),
            '0.73': (12, 864, 1184),
            '0.78': (13, 896, 1152),  # 3:4
            '0.83': (14, 928, 1120),
            '0.88': (15, 960, 1088),
            '0.94': (16, 992, 1056),
            '1.0': (17, 1024, 1024),
            '1.06': (18, 1056, 992),
            '1.13': (19, 1088, 960),
            '1.21': (20, 1120, 928),
            '1.29': (21, 1152, 896),  # 4:3
            '1.37': (22, 1184, 864),
            '1.46': (23, 1216, 832),
            '1.5': (24, 1248, 832),  # 3:2
            '1.6': (25, 1280, 800),
            '1.71': (26, 1312, 768),
            '1.75': (27, 1344, 768),  # 16：9
            '1.87': (28, 1376, 736),
            '1.91': (29, 1408, 736),  # 1.91:1
            '2.05': (30, 1440, 704),
            '2.09': (31, 1472, 704),
            '2.24': (32, 1504, 672),
            '2.29': (33, 1536, 672)
        }

    def get_image_hash(self, image):
        m = hashlib.md5()
        m.update(image)
        return m.hexdigest()

    def estimate_jpeg_quality(self, img_bytes):

        try:
            # with open(path, 'rb') as f:
            #     data = f.read()

            # 找到 DQT 段（量化表标记为 0xFFDB）
            pos = img_bytes.find(b'\xFF\xDB')
            if pos == -1:
                return None

            qtable = img_bytes[pos + 5:pos + 5 + 64]
            if len(qtable) < 64:
                return None

            std_table = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ])
            qtable = np.frombuffer(qtable, dtype=np.uint8).reshape((8, 8))
            scale = np.mean(std_table / qtable)
            quality = round(scale * 50)
            return int(np.clip(quality, 1, 100))

        except Exception as e:
            return None


    # Laplacian 方差（Variance of Laplacian）, 方差越大 → 图像越清晰；小于 100 通常认为模糊。
    def calculate_sharpness_laplacian(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            return float(lap.var())
        except:
            return -1.0

    # 梯度越大 → 边缘越强，代表越锐利
    def edge_clarity_sobel(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
            clarity_score = np.mean(sobel)
            return float(clarity_score)
        except:
            return -1.0

    # 平均亮度（Brightness）
    def calculate_weight_brightness(self, image):
        try:
            b, g, r = cv2.split(image)
            brightness_weighted = np.mean(0.114 * b + 0.587 * g + 0.299 * r)
            return float(brightness_weighted)
        except:
            return -1.0

    # 平均亮度（Brightness）
    def calculate_avg_brightness(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            return float(brightness)
        except:
            return -1.0

    def calculate_contrast(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return float(np.std(gray))
        except:
            return -1.0

    # 欠曝/过曝像素比例
    def calculate_exposure_ratio(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            total = gray.size
            under_exposed = float(np.sum(hist[:40]) / total)
            over_exposed = float(np.sum(hist[215:]) / total)
            return under_exposed, over_exposed
        except:
            return -1.0, -1.0

    # 局部噪声估计（滑动窗口方差）, 一个干净且细节丰富的图像，局部噪声标准差可能在 0-10 左右,
    # 一个明显有噪点的图像，局部噪声标准差可能超过 20-30。
    # 纯黑或纯白区域噪声接近 0。
    def estimate_noise(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            noise = []
            for i in range(0, h - 8, 8):
                for j in range(0, w - 8, 8):
                    patch = gray[i:i + 8, j:j + 8]
                    noise.append(np.var(patch))
            return float(np.mean(noise))
        except:
            return -1.0

    # bpp > 1.5 → 较低压缩（高质量）, bpp < 0.5 → 高压缩（低质量）
    def estimate_jpg_compression(self, image, img_file_size):
        try:
            h, w, c = image.shape
            bpp = float(img_file_size) / float(h * w)
            return bpp, h, w
        except:
            return -1.0, -1, -1

    def resize_max_edge_lanczos(self, img, max_size=1024, min_size=1024):
        try:
            h, w = img.shape[:2]

            # scale = max_size / max(h, w)
            scale = min_size/min(h, w)

            # 如果图片已经比 max_size 小，就不缩放
            if scale >= 1:
                return img, h, w

            new_w = int(w * scale)
            new_h = int(h * scale)

            # 使用 Lanczos 插值
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        except Exception as e:
            resized = None
            new_h = -1
            new_w = -1

        return resized, new_h, new_w

    def resize_to_bucket_lanczos(self, img):
        try:
            h, w = img.shape[:2]

            # which bucket
            aspect_ratio = float(w)/h
            target_bucket_id = -1
            abs_distance = 1
            target_w = w
            target_h = h
            for ar in self.bucket_info.keys():

                bucket_id, bucket_w, bucket_h = self.bucket_info[ar]
                ar = float(ar)

                if abs(ar-aspect_ratio) < abs_distance:
                    abs_distance = abs(ar-aspect_ratio)
                    target_w = bucket_w
                    target_h = bucket_h
                    target_bucket_id = bucket_id

            # 使用 Lanczos 插值
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        except Exception as e:
            print(e)
            resized = None
            target_h = -1
            target_w = -1
            target_bucket_id = -1

        return resized, target_h, target_w, target_bucket_id


    def get_image_info_from_base64(self, base64_str, img_path):

        try:
            if base64_str is not None:
                b64_string = base64_str.strip().replace('\n', '').replace('\r', '')
                padding = b64_string.count('=')
                length = len(b64_string)
                img_file_size = length * 3 // 4 - padding
                # print('base64_str img_file_size:', img_file_size)

                # 解码 base64 为字节
                img_data = base64.b64decode(base64_str)

            elif img_path is not None:
                img_file_size = os.path.getsize(img_path)
                with open(img_path, "rb") as f:
                    img_data = f.read()
                # print('img_path img_file_size:', img_file_size)

            else:
                img_file_size = -1
                assert img_path is not None or base64_str is not None

            img_quality = self.estimate_jpeg_quality(img_data)
            # 转换为 NumPy 数组
            np_arr = np.frombuffer(img_data, np.uint8)

            # 用 OpenCV 解码为图像
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img, img_file_size, img_quality

        except Exception as e:
            # print(e)
            return None, -1, -1

    def Process(self, inputrow, outputrow):

        outputrow["ImageMD5"] = inputrow["ImageMD5"]
        if inputrow.__contains__("Base64Data") and inputrow["Base64Data"] is not None:
            img_base64 = inputrow["Base64Data"]
            image, img_file_size, img_quality = self.get_image_info_from_base64(base64_str = img_base64, img_path = None)
        else:
            img_path = inputrow["ImagePath"]
            image, img_file_size, img_quality = self.get_image_info_from_base64(base64_str = None, img_path = img_path)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        encode_param_v2 = [int(cv2.IMWRITE_JPEG_QUALITY), 85]


        bpp, height, width = self.estimate_jpg_compression(image, img_file_size)
        aspect_ratio = float(width)/height

        # compression_ratio
        _, encimg_raw = cv2.imencode('.jpg', image, encode_param)
        compression_ratio = float(img_file_size)/len(encimg_raw)

        # print('image:', image.shape)
        resized_image, resized_h, resized_w, target_bucket_id = self.resize_to_bucket_lanczos(image)

        img_quality_ratio = img_quality
        if img_quality == None:
            img_quality_ratio = 100
            img_quality = -1

        encode_param_resize = [int(cv2.IMWRITE_JPEG_QUALITY), img_quality_ratio]
        _, encimg_resized = cv2.imencode('.jpg', resized_image, encode_param_resize)
        _, encimg_resized_v2 = cv2.imencode('.jpg', resized_image, encode_param_v2)
        img_file_size_resized = len(encimg_resized)
        bpp_resized = img_file_size_resized / float(resized_h * resized_w)
        bpp_resized_v2 = len(encimg_resized_v2) / float(resized_h * resized_w)

        # image = resized_image

        # print(memory_bytes, resized_h, resized_w, target_bucket_id, memory_bytes/(resized_h*resized_w))
        # bpp, height, width = self.estimate_jpg_compression(image, img_file_size)

        laplacian_clarity = self.calculate_sharpness_laplacian(image)
        edge_clarity = self.edge_clarity_sobel(image)
        img_contrast = self.calculate_contrast(image)

        under_exposed, over_exposed = self.calculate_exposure_ratio(image)
        weight_brightness = self.calculate_weight_brightness(image)
        avg_brightness = self.calculate_avg_brightness(image)
        img_noise = self.estimate_noise(image)

        outputrow["width"] = width
        outputrow["height"] = height
        outputrow["width_resized"] = resized_w
        outputrow["height_resized"] = resized_h
        outputrow["aspect_ratio"] = aspect_ratio
        outputrow["img_bytes"] = img_file_size
        outputrow["img_bytes_resized"] = img_file_size_resized
        outputrow["bpp"] = bpp
        outputrow["bpp_resized"] = bpp_resized
        outputrow["bpp_resized_v2"] = bpp_resized_v2
        outputrow["laplacian_clarity"] = laplacian_clarity
        outputrow["edge_clarity"] = edge_clarity
        outputrow["img_contrast"] = img_contrast
        outputrow["under_exposed"] = under_exposed
        outputrow["over_exposed"] = over_exposed
        outputrow["img_noise"] = img_noise

        outputrow["weight_brightness"] = weight_brightness
        outputrow["avg_brightness"] = avg_brightness

        outputrow["target_bucket_id"] = target_bucket_id
        outputrow["img_quality"] = img_quality
        outputrow["compression_ratio"] = compression_ratio

        if img_file_size < 0 or laplacian_clarity < 0:
            outputrow["succeed"] = -1
        else:
            outputrow["succeed"] = 1

        return outputrow

def test_by_single_image(img_path, is_base64):
    SimpleImageQualityTest = SimpleImageQuality()
    if is_base64:
        # # 读取 JPG 文件
        with open(img_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        inputrow = {
            'Base64Data': img_base64,
            'ImageMD5': img_path,
            'ImagePath': img_path,
        }
    else:
        inputrow = {
            'Base64Data': None,
            'ImageMD5': img_path,
            'ImagePath': img_path,
        }

    outputrow = {}
    res = SimpleImageQualityTest.Process(inputrow, outputrow)
    return res


def process_image_by_batch(batch_items, splits=0):
    SimpleImageQualityTest = SimpleImageQuality()

    results = []
    for inputrow in batch_items:
        outputrow = {}
        res = SimpleImageQualityTest.Process(inputrow, outputrow)
        res_list = [str(v) for v in res.values()]
        results.append('\t'.join(res_list))
    return results    

if __name__ == "__main__":
    # ====== 1. 命令行参数 ======
    parser = argparse.ArgumentParser(description="Parallel image batch processor")
    parser.add_argument("--input", required=True, help="Path to input TSV file")
    parser.add_argument("--output", required=True, help="Directory to save results")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of parallel tasks")
    args = parser.parse_args()

    in_file = Path(args.input)
    out_dir = Path(args.output)
    task_nums = args.num_tasks
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== 2. 读取输入文件 ======
    with open(in_file, 'r', encoding='utf-8') as f:
        fr = f.readlines()
    total_num = len(fr)
    print(f"📘 Total lines: {total_num}")

    # ====== 3. 拆分任务 ======
    num_of_each_split = total_num // task_nums
    task_list = []

    for i in range(task_nums):
        s_index = i * num_of_each_split
        # ✅ 最后一段取到结尾，避免遗漏
        e_index = (s_index + num_of_each_split) if i < task_nums - 1 else total_num
        lines = fr[s_index:e_index]

        img_list = []
        for line in lines:
            # 假设分隔符是多个空格
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            id, img_path = parts[:2]
            inputrow = {
                'Base64Data': None,
                'ImageMD5': id,
                'ImagePath': img_path,
            }
            img_list.append(inputrow)
        task_list.append(img_list)

    print(f"✅ Total tasks: {len(task_list)}, example batch size: {len(task_list[0])}")

    # ====== 4. 并行执行 ======
    all_results = Parallel(n_jobs=task_nums, backend='loky')(
        delayed(process_image_by_batch)(task_list[i], i, out_dir)
        for i in range(task_nums)
    )

    # === 主进程合并保存 ===
    out_file = out_dir / "output.txt"
    with open(out_file, 'w', encoding='utf-8') as fw:
        for task_res in all_results:
            for line in task_res:
                fw.write(line + '\n')

    print(f"✅ Done! Merged output saved to {out_file}")

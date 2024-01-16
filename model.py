import numpy as np
import time
import matplotlib.pyplot as plt


class Model:
    def __init__(self, img_tf):
        self.patch_size = (9, 9)
        self.img_tf = img_tf / 255
        self.stride = 4

    def extract_patches(self):  # 图像，patch 大小(元组)，patch 间重叠的行/列数
        height, width = self.img_tf.shape
        patch_height, patch_width = self.patch_size

        # 计算每一维度的 patch 个数
        num_patches_h = (height - patch_height) // self.stride + 1
        num_patches_w = (width - patch_width) // self.stride + 1

        # 初始化输出向量
        num_patches = num_patches_h * num_patches_w
        patches = np.zeros((num_patches, patch_height, patch_width))

        # 遍历
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # 获取 patch 索引
                k = i * num_patches_w + j
                # 获取 patch 坐标
                x = i * self.stride
                y = j * self.stride
                # 从图像中提取 patch
                patches[k] = self.img_tf[x : x + patch_height, y : y + patch_width]

        self.X = patches
        print("提取patch OK...")

    def svd_decomposition(self):  # 这里必须使用这个函数对生成的奇异值矩阵进行 reshape，否则 S 的形状会有问题
        # 移除 X 的 DC 部分
        self.X = self.X.reshape(self.X.shape[0], -1)
        self.X -= np.mean(self.X, axis=0)
        self.X /= np.std(self.X, axis=0)
        self.X = self.X.T

        self.D0, S, V = np.linalg.svd(self.X, full_matrices=True)
        S = np.diag(S)
        zeros = np.zeros((S.shape[0], V.shape[1] - self.D0.shape[0]))
        S = np.hstack([S, zeros])
        self.A0 = S @ V.T
        print("SVD分解 OK...")

    def plot(self, data):  # 绘制字典图像
        # 创建一个 10x10 的画布
        plt.figure("字典图像", figsize=data.shape[0:2])
        # 显示子图
        plt.imshow(data, cmap="viridis")
        # 隐藏坐标轴
        plt.axis("on")
        # 图像标题
        plt.title("Dictionary")
        # 显示画布
        plt.show()

    def iteration(self):
        D = self.D0
        A = self.A0
        lambd = 50
        maxits = 1e4
        tol = 1e-4
        ek = np.zeros(int(maxits))  # 残差矩阵

        start = time.time()  # 初始化计时器
        its = 0  # 迭代计数器
        while its < maxits - 1:
            D_old = D.copy()
            A_old = A.copy()

            # 计算 D 的梯度
            L_D = np.linalg.norm(A @ A.T) + 0.1  # 估计李普希策常数
            gamma_D = 1.9 / L_D  # 下降步长

            grad_D = (-self.X + D @ A) @ A.T  # D 梯度
            D = D - gamma_D * grad_D  # 更新 D 矩阵

            # 对原子执行单位长度限制
            for j in range(81):
                D[:, j] = D[:, j] / np.linalg.norm(D[:, j])

            # 计算 A 的梯度
            L_A = np.linalg.norm(D.T @ D) + 0.1  # 估计李普希策常数
            gamma_A = 1.9 / L_A  # 步长

            grad_A = D.T @ (-self.X + D @ A)  # A 梯度
            w = A - gamma_A * grad_A  # 更新 A 矩阵

            # l1-范数的软阈值收缩
            A = np.sign(w) * np.maximum(np.abs(w) - lambd * gamma_A, 0)

            ek[its] = np.linalg.norm(D - D_old) / np.linalg.norm(D)  # 计算残差

            if its % 500 == 0:  # 每 500 次迭代更新一次 lambda
                lambd = max(lambd / 1.5, 0.5)

            # 下降停止的条件
            if ek[its] < tol or ek[its] > 1e10:
                if its > 1000 - 1:
                    break

            if its % 500 == 0:  # 每 500 次迭代打印一次信息
                print(
                    f"{its:06d}, time: {time.time() - start:05.2f}s, rel_error: {ek[its]:05.2e}, obj_value: {np.linalg.norm(self.X - D @ A, 'fro')**2:05.2e}"
                )

            its += 1  # 迭代计数 +1
        print(
            f"{its:06d}, final iteration time: {time.time() - start:05.2f}s, rel_error: {ek[its]:05.2e}, obj_value: {np.linalg.norm(self.X - D @ A, 'fro')**2:05.2e}"
        )
        print("\n")

        ek = ek[:its]  # 将 ek 截断为实际迭代次数

        i_size = np.prod(self.patch_size) + self.patch_size[0]  # 计算字典矩阵大小
        self.imD = np.zeros((i_size, i_size))  # 初始化字典矩阵

        c = 0
        for i in range(self.patch_size[0]):  # 遍历行
            range_i = slice(
                i * (self.patch_size[0] + 1), (i + 1) * (self.patch_size[0] + 1) - 1
            )  # define the row range
            for j in range(self.patch_size[1]):  # 遍历列
                c += 1
                range_j = slice(
                    j * (self.patch_size[1] + 1), (j + 1) * (self.patch_size[1] + 1) - 1
                )

                atom = D[:, c - 1]  # 从 D 矩阵中取得原子
                self.imD[range_i, range_j] = atom.reshape(self.patch_size) - np.min(
                    atom
                )  # Reshape原子并提取最小值

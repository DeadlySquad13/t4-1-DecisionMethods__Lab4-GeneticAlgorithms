var sensorWeb = [[-250, -445], [-200, -445], [-150, -445], [-100, -445], [-50, -445], [0, -445], [50, -445], [100, -445], [150, -445], [200, -445], [-250, -370], [-200, -370], [-150, -370], [-100, -370], [-50, -370], [0, -370], [50, -370], [100, -370], [150, -370], [200, -370], [-250, -295], [-200, -295], [-150, -295], [-100, -295], [-50, -295], [0, -295], [50, -295], [100, -295], [150, -295], [200, -295], [-250, -220], [-200, -220], [-150, -220], [-100, -220], [-50, -220], [0, -220], [50, -220], [100, -220], [150, -220], [200, -220], [-250, -145], [-200, -145], [-150, -145], [-100, -145], [-50, -145], [0, -145], [50, -145], [100, -145], [150, -145], [200, -145], [-250, -70], [-200, -70], [-150, -70], [-100, -70], [-50, -70], [0, -70], [50, -70], [100, -70], [150, -70], [200, -70], [-250, 5], [-200, 5], [-150, 5], [-100, 5], [-50, 5], [0, 5], [50, 5], [100, 5], [150, 5], [200, 5], [-250, 80], [-200, 80], [-150, 80], [-100, 80], [-50, 80], [0, 80], [50, 80], [100, 80], [150, 80], [200, 80], [-250, 155], [-200, 155], [-150, 155], [-100, 155], [-50, 155], [0, 155], [50, 155], [100, 155], [150, 155], [200, 155], [-250, 230], [-200, 230], [-150, 230], [-100, 230], [-50, 230], [0, 230], [50, 230], [100, 230], [150, 230], [200, 230], [-250, 305], [-200, 305], [-150, 305], [-100, 305], [-50, 305], [0, 305], [50, 305], [100, 305], [150, 305], [200, 305], [-250, 380], [-200, 380], [-150, 380], [-100, 380], [-50, 380], [0, 380], [50, 380], [100, 380], [150, 380], [200, 380]];

var W = [[0.98, 0.03, -1.24, 0.84, -0.53], [-0.04, 0.29, 0.93, 0.75, -0.16], [0.31, 0.14, 2.58, 0.18, 1.22], [0.43, -0.84, -0.02, -1.15, 0.64], [0.22, 0.25, -1.04, 0.8, 0.0], [-0.49, -1.04, 0.48, 1.06, -1.24], [-1.2, 0.0, -1.48, -1.98, 1.29], [-1.62, -0.16, -0.81, 0.13, -0.11], [-1.25, 0.27, -0.19, -1.77, -0.19], [-0.38, -0.07, -0.79, -0.34, 0.12], [0.21, 0.36, 1.01, 0.01, 0.21], [1.47, 0.97, -0.01, 0.1, 0.83], [0.91, 0.49, 1.81, 0.52, 1.58], [-0.2, 0.01, -0.45, 2.22, 2.13], [1.73, -0.71, -0.15, -1.54, -0.46], [-0.79, 0.46, 1.6, -0.05, -1.02], [-0.54, 1.49, 0.12, -0.56, 0.43], [-0.18, 0.89, -1.28, 2.22, -0.53], [-0.91, -0.99, -1.58, 0.19, 1.69], [0.96, 1.0, -1.18, -1.38, -1.24], [0.55, 0.12, 0.08, 1.25, -0.92], [0.14, 0.74, -0.81, -0.17, 1.73], [1.11, -0.02, 0.35, -1.68, 0.06], [0.8, 0.07, -1.23, -0.2, -0.38], [0.3, -1.32, -0.48, -1.61, 1.02], [0.4, 0.09, 1.67, 0.23, 0.68], [0.02, 0.45, 0.92, -2.06, -0.8], [1.78, 0.68, 0.41, 1.03, 0.47], [-0.52, 0.81, 1.2, -0.6, 0.99], [0.67, -2.35, -1.44, 0.17, 1.33], [0.98, -1.49, 2.28, -1.37, 0.59], [-0.81, -0.84, -2.24, 1.16, 0.26], [0.66, -0.28, 0.84, -0.08, 0.18], [-0.55, 0.98, -0.94, 0.28, -1.23], [1.37, -1.38, 0.24, -0.71, 0.74], [-0.68, -0.73, -0.9, 0.59, -0.49], [-0.23, -1.68, 0.47, 0.61, 0.24], [-0.15, -1.13, -0.04, -0.57, 0.44], [-1.22, 0.94, -0.52, -0.9, -0.23], [-0.57, -1.41, 0.0, -2.3, 1.68], [-0.65, 1.49, -0.81, 0.09, 0.52], [-0.39, -0.63, -0.2, -1.77, 0.3], [-1.09, 0.13, 0.73, 1.25, -0.59], [0.26, 0.99, -1.11, -1.84, 1.29], [0.28, -1.34, 0.17, -0.57, 2.83], [-0.01, -0.93, -0.14, 1.47, 0.18], [-1.39, -1.18, -0.06, 2.15, -1.07], [1.15, 1.51, -0.36, -1.3, -1.65], [-1.58, 0.79, 0.42, 0.0, -0.79], [-0.85, -0.16, -0.95, 0.54, 1.9], [0.19, 1.97, -0.08, -0.16, 0.59], [-0.07, -0.23, -0.38, 0.29, -1.22], [-0.98, 2.27, -0.91, 0.09, -0.91], [1.95, -1.49, -1.96, 1.2, -0.53], [0.36, 0.15, -0.79, -0.52, 0.38], [1.01, -0.18, -1.14, 0.28, -2.51], [-0.73, -0.69, 0.26, 0.16, -1.61], [0.13, 0.32, 0.51, 0.19, -0.92], [0.53, 0.21, 0.52, -0.38, -0.18], [-0.07, -0.37, -0.11, -0.24, 1.28], [-0.96, -0.5, 0.08, 0.17, 0.21], [0.85, -1.28, -0.31, -0.19, 0.57], [0.5, -0.61, 0.82, -1.31, 0.45], [0.92, -1.97, -0.57, -0.64, 0.37], [0.52, -0.54, -1.37, -1.39, 1.15], [0.26, -0.95, 2.03, 1.15, 1.37], [0.44, 0.11, -0.78, -0.97, -1.11], [0.13, 0.4, 0.57, 0.61, -1.82], [-1.86, 0.76, -0.95, -1.4, 1.88], [0.75, -0.17, 0.87, 0.35, 0.96], [-1.53, -2.12, -0.24, 0.25, 1.11], [1.46, 1.42, 1.27, 1.57, -2.44], [-0.05, 0.44, -0.2, -0.31, -1.65], [0.48, 1.08, -0.39, -0.25, 0.15], [0.49, 1.39, -1.1, -1.85, -1.66], [0.04, -0.63, -1.24, -1.2, 0.28], [1.81, -1.0, 0.39, 3.11, 1.23], [0.44, -1.88, 1.05, -0.18, 1.63], [-1.01, 0.05, -0.92, -0.66, 0.15], [1.48, 0.52, 0.4, -1.57, -0.57], [-1.45, -0.19, -0.71, -0.4, -1.74], [-1.0, 0.08, 2.37, 0.19, 0.02], [0.09, -1.59, -0.36, 0.49, 0.18], [0.69, 0.51, -1.98, 0.36, -0.87], [0.49, -0.34, 0.54, 0.3, -1.15], [0.37, 0.7, -2.25, 0.9, -0.62], [-0.38, -0.24, 0.14, 0.12, -0.69], [0.31, -1.11, 1.51, 0.93, 0.8], [-1.97, -1.0, -0.94, 0.2, 1.13], [-2.49, 0.54, -2.46, 0.45, 0.1], [0.17, -0.59, 0.73, -0.75, 1.78], [0.24, 0.04, 0.04, -1.42, -0.43], [-0.98, -1.15, 0.07, -0.51, -1.52], [1.2, -0.34, 0.48, -1.43, -0.09], [2.22, -0.34, -0.89, -1.1, 0.14], [-1.11, -0.52, -0.75, 1.3, -0.27], [0.16, -0.3, -0.23, -2.13, -0.62], [-0.97, -0.89, -0.4, 0.91, -1.08], [1.83, -0.51, 0.97, -0.03, 0.2], [-0.16, 0.13, 1.49, 1.78, 0.2], [-0.73, 0.17, 1.75, -2.9, 0.62], [0.63, 0.46, 1.44, 0.0, 0.93], [0.95, 2.11, 0.19, -0.7, -0.79], [0.48, 0.1, -0.37, -0.81, -0.98], [0.19, -0.7, 0.05, 0.19, 1.97], [0.99, -0.16, -0.9, -1.77, 0.84], [-0.87, -0.09, 1.31, 0.11, 0.48], [1.11, -1.18, -0.52, -0.61, 0.91], [0.1, -0.07, 1.39, 2.62, 1.51], [-1.31, -0.5, 0.05, -0.01, -0.32], [-1.1, -0.17, -0.16, -0.88, -0.44], [-1.9, 0.04, 0.05, 1.52, 2.9], [1.16, -0.02, 0.77, -0.97, -0.08], [1.14, -2.05, -0.4, -0.65, -0.02], [0.55, 0.37, 0.87, -1.48, -0.16], [-1.28, 0.35, 1.01, -0.46, -0.63], [1.17, 1.26, -2.69, -0.34, -0.02], [0.63, -0.55, -0.15, -1.23, 1.4], [2.18, 0.91, 2.43, 1.24, 1.19], [-0.49, 0.48, 0.81, 1.08, 0.84], [-0.41, 0.28, -0.71, 1.57, -1.35], [-0.17, 1.18, 0.77, -0.18, 1.29], [1.05, -2.29, 0.41, -1.57, -0.57]];

var W2 = [[1.16, 0.08, -0.31], [-0.37, -0.61, -0.3], [-1.19, 0.83, -0.6], [-0.16, -0.15, -0.01], [-0.19, 1.11, -0.16]];
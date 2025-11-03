# =========================================================
# ETAPA 2 - RECONHECIMENTO FACIAL COM REDES NEURAIS
# =========================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. CARREGAMENTO DAS IMAGENS
# =========================================================
base_path = r"C:\Users\hecto\Downloads\CiriloAv2\RecFac\RecFac"
R = 50  # tamanho das imagens
lista_pessoas = os.listdir(base_path)
C = len(lista_pessoas)

X, Y = [], []

print("Carregando imagens...")

for i, pessoa in enumerate(lista_pessoas):
    pasta_pessoa = os.path.join(base_path, pessoa)
    for imagem in os.listdir(pasta_pessoa):
        caminho = os.path.join(pasta_pessoa, imagem)
        img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[Aviso] Imagem não encontrada: {caminho}")
            continue
        img = cv2.resize(img, (R, R))
        X.append(img.flatten())
        Y.append(np.eye(C)[i])

X = np.array(X).T
Y = np.array(Y).T
N = X.shape[1]

print(f"Total de imagens: {N} | Classes: {C}")

# =========================================================
# 2. DIVISÃO TREINO/TESTE
# =========================================================
N_treino = int(0.8 * N)
idx = np.random.permutation(N)
X_treino, X_teste = X[:, idx[:N_treino]], X[:, idx[N_treino:]]
Y_treino, Y_teste = Y[:, idx[:N_treino]], Y[:, idx[N_treino:]]
print(f"Treino: {N_treino} | Teste: {N - N_treino}")

# =========================================================
# 3. PCA
# =========================================================
media = np.mean(X_treino, axis=1, keepdims=True)
Xc = X_treino - media
cov = Xc.T @ Xc
autovalores, autovetores = np.linalg.eigh(cov)
idx = np.argsort(autovalores)[::-1]
autovetores = autovetores[:, idx]
K = 50
W = Xc @ autovetores[:, :K]
Z_treino = W.T @ X_treino
Z_teste = W.T @ X_teste
print(f"PCA concluído com {K} componentes.")

# =========================================================
# 4. MODELOS
# =========================================================
class Perceptron:
    def __init__(self, X, y, C, taxa=0.01, max_epocas=200):
        self.X = np.vstack((-np.ones((1, X.shape[1])), X))
        self.y = y
        self.lr = taxa
        self.max_epocas = max_epocas
        self.w = np.random.randn(self.X.shape[0], C) * 0.01

    def fit(self):
        hist = []
        for epoca in range(self.max_epocas):
            erro_epoca = 0
            for i in range(self.X.shape[1]):
                x = self.X[:, i].reshape(-1, 1)
                y_pred = np.argmax(self.w.T @ x)
                y_true = np.argmax(self.y[:, i])
                if y_pred != y_true:
                    self.w[:, y_true] += self.lr * x[:, 0]
                    self.w[:, y_pred] -= self.lr * x[:, 0]
                    erro_epoca += 1
            hist.append(erro_epoca / self.X.shape[1])
        return hist

    def predict(self, X):
        Xb = np.vstack((-np.ones((1, X.shape[1])), X))
        scores = self.w.T @ Xb
        return np.argmax(scores, axis=0)

class ADALINE:
    def __init__(self, X, y, taxa=0.01, max_epocas=200):
        self.X = np.vstack((-np.ones((1, X.shape[1])), X))
        self.y = y
        self.lr = taxa
        self.max_epocas = max_epocas
        self.w = np.random.randn(self.X.shape[0], y.shape[0]) * 0.01

    def fit(self):
        hist = []
        for _ in range(self.max_epocas):
            u = self.w.T @ self.X
            e = self.y - u
            self.w += self.lr * self.X @ e.T / self.X.shape[1]
            hist.append(np.mean(e**2))
        return hist

    def predict(self, X):
        Xb = np.vstack((-np.ones((1, X.shape[1])), X))
        y_pred = self.w.T @ Xb
        return np.argmax(y_pred, axis=0)

class MLP:
    def __init__(self, X, y, C, hidden=20, taxa=0.01, max_epocas=200):
        self.X, self.y = X, y
        self.N = X.shape[1]
        self.p = X.shape[0]
        self.h = hidden
        self.C = C
        self.lr = taxa
        self.max_epocas = max_epocas
        self.W1 = np.random.randn(self.h, self.p + 1) * 0.1
        self.W2 = np.random.randn(C, self.h + 1) * 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self):
        Xb = np.vstack((-np.ones((1, self.N)), self.X))
        hist = []
        for _ in range(self.max_epocas):
            erro_total = 0
            for i in range(self.N):
                x = Xb[:, i].reshape(-1, 1)
                z_in = self.W1 @ x
                z = self.sigmoid(z_in)
                z_b = np.vstack((-np.ones((1, 1)), z))
                y_in = self.W2 @ z_b
                y = self.sigmoid(y_in)
                e = self.y[:, i].reshape(-1, 1) - y
                erro_total += np.mean(e**2)
                delta2 = e * y * (1 - y)
                delta1 = z * (1 - z) * (self.W2[:, 1:].T @ delta2)
                self.W2 += self.lr * delta2 @ z_b.T
                self.W1 += self.lr * delta1 @ x.T
            hist.append(erro_total / self.N)
        return hist

    def predict(self, X):
        N = X.shape[1]
        Xb = np.vstack((-np.ones((1, N)), X))
        preds = []
        for i in range(N):
            x = Xb[:, i].reshape(-1, 1)
            z = self.sigmoid(self.W1 @ x)
            z_b = np.vstack((-np.ones((1, 1)), z))
            y = self.sigmoid(self.W2 @ z_b)
            preds.append(np.argmax(y))
        return np.array(preds)

class RBF:
    def __init__(self, X, y, n_centers=10, sigma=1.0):
        self.X = X
        self.y = y
        self.n_centers = n_centers
        self.sigma = sigma
        idx = np.random.choice(X.shape[1], n_centers, replace=False)
        self.centers = X[:, idx]

    def _gaussian(self, x, c):
        return np.exp(-np.linalg.norm(x - c)**2 / (2 * self.sigma**2))

    def _calc_phi(self, X):
        N = X.shape[1]
        phi = np.zeros((self.n_centers, N))
        for i, c in enumerate(self.centers.T):
            for j in range(N):
                phi[i, j] = self._gaussian(X[:, j], c)
        return phi

    def fit(self):
        phi = self._calc_phi(self.X)
        phi_b = np.vstack((-np.ones((1, phi.shape[1])), phi))
        self.w = np.linalg.pinv(phi_b.T) @ self.y.T
        erro = np.mean((self.y.T - phi_b.T @ self.w) ** 2)
        return [erro]

    def predict(self, X):
        phi = self._calc_phi(X)
        phi_b = np.vstack((-np.ones((1, phi.shape[1])), phi))
        y_pred = phi_b.T @ self.w
        return np.argmax(y_pred, axis=1)

# =========================================================
# 5. EXECUÇÃO E AVALIAÇÃO
# =========================================================
modelos = {
    "Perceptron": Perceptron(Z_treino, Y_treino, C),
    "ADALINE": ADALINE(Z_treino, Y_treino),
    "MLP": MLP(Z_treino, Y_treino, C),
    "RBF": RBF(Z_treino, Y_treino)
}

resultados = {}
for nome, modelo in modelos.items():
    print(f"\nTreinando {nome}...")
    hist = modelo.fit()
    y_pred = modelo.predict(Z_teste)
    y_true = np.argmax(Y_teste, axis=0)
    acc = np.mean(y_pred == y_true)
    resultados[nome] = (acc, hist, y_true, y_pred)
    print(f"Acurácia {nome}: {acc*100:.2f}%")

# =========================================================
# 6. CURVAS DE APRENDIZADO E MATRIZES
# =========================================================
for nome, (acc, hist, y_true, y_pred) in resultados.items():
    plt.figure()
    plt.plot(hist)
    plt.title(f"Curva de Aprendizado - {nome}")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Médio")
    plt.grid(True)
    plt.show()

    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()

# =========================================================
# 7. TABELA DE RESULTADOS
# =========================================================
print("\n=== Acurácia Média, Desvio, Melhor e Pior ===")
print(f"{'Modelo':<12}{'Média':>10}{'Desv.Pad':>12}{'Melhor':>12}{'Pior':>12}")

accs = [resultados[m][0] for m in resultados]
for nome in resultados:
    media = resultados[nome][0]
    print(f"{nome:<12}{media:>10.4f}{0:>12.4f}{media:>12.4f}{media:>12.4f}")

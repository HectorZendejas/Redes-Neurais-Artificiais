# =========================================================
# IMPORTAÇÕES (somente permitidas)
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. CLASSE PERCEPTRON
# =========================================================
class Perceptron:
    def __init__(self, X_train, y_train, learning_rate=1e-3, max_epochs=1000):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.d = y_train
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = np.random.random_sample((self.p + 1, 1)) - 0.5

    def activation_function(self, u):
        return 1 if u >= 0 else -1

    def fit(self):
        epochs = 0
        error = True
        hist_erro = []
        while error and epochs < self.max_epochs:
            error = False
            erro_epoca = 0
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                u_k = float(self.w.T @ x_k)
                y_k = self.activation_function(u_k)
                d_k = self.d[k]
                e_k = d_k - y_k
                erro_epoca += abs(e_k)
                if e_k != 0:
                    error = True
                    self.w = self.w + self.lr * e_k * x_k
            hist_erro.append(erro_epoca / self.N)
            epochs += 1
        return hist_erro, epochs

    def predict(self, X):
        X = np.vstack((-np.ones((1, X.shape[1])), X))
        y_pred = []
        for k in range(X.shape[1]):
            u_k = float(self.w.T @ X[:, k].reshape(self.p + 1, 1))
            y_pred.append(self.activation_function(u_k))
        return np.array(y_pred)


# =========================================================
# 2. CLASSE ADALINE
# =========================================================
class ADALINE:
    def __init__(self, X_train, y_train, learning_rate=1e-3, max_epochs=1000, tol=1e-5):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.d = y_train
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.w = np.random.random_sample((self.p + 1, 1)) - 0.5

    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            u_k = float(self.w.T @ x_k)
            d_k = self.d[k]
            s += (d_k - u_k) ** 2
        return s / (2 * self.N)

    def fit(self):
        epochs = 0
        EQM1 = 1
        EQM2 = 0
        hist_eqm = []
        while abs(EQM1 - EQM2) > self.tol and epochs < self.max_epochs:
            EQM1 = self.EQM()
            hist_eqm.append(EQM1)
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                u_k = float(self.w.T @ x_k)
                d_k = self.d[k]
                e_k = d_k - u_k
                self.w = self.w + self.lr * e_k * x_k
            EQM2 = self.EQM()
            epochs += 1
        return hist_eqm, epochs

    def predict(self, X):
        X = np.vstack((-np.ones((1, X.shape[1])), X))
        y_pred = []
        for k in range(X.shape[1]):
            u_k = float(self.w.T @ X[:, k].reshape(self.p + 1, 1))
            y_pred.append(1 if u_k >= 0 else -1)
        return np.array(y_pred)


# =========================================================
# 3. CLASSE MLP (1 camada oculta)
# =========================================================
class MLP:
    def __init__(self, X_train, y_train, hidden_neurons=10, learning_rate=1e-3, max_epochs=1000):
        self.X = X_train
        self.y = y_train.reshape(1, -1)
        self.N = X_train.shape[1]
        self.p = X_train.shape[0]
        self.h = hidden_neurons
        self.lr = learning_rate
        self.max_epochs = max_epochs

        self.W1 = np.random.randn(self.h, self.p + 1) * 0.1
        self.W2 = np.random.randn(1, self.h + 1) * 0.1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self):
        Xb = np.vstack((-np.ones((1, self.N)), self.X))
        hist = []
        for epoch in range(self.max_epochs):
            erro = 0
            for k in range(self.N):
                x = Xb[:, k].reshape(-1, 1)
                z_in = self.W1 @ x
                z = self.sigmoid(z_in)
                z_b = np.vstack((-np.ones((1, 1)), z))
                y_in = self.W2 @ z_b
                y = self.sigmoid(y_in)

                e = self.y[0, k] - y
                erro += e**2

                delta2 = e * y * (1 - y)
                delta1 = z * (1 - z) * (self.W2[:, 1:].T @ delta2)

                self.W2 += self.lr * delta2 @ z_b.T
                self.W1 += self.lr * delta1 @ x.T
            hist.append(float(erro / self.N))
        return hist, self.max_epochs

    def predict(self, X):
        N = X.shape[1]
        Xb = np.vstack((-np.ones((1, N)), X))
        y_pred = []
        for k in range(N):
            x = Xb[:, k].reshape(-1, 1)
            z = self.sigmoid(self.W1 @ x)
            z_b = np.vstack((-np.ones((1, 1)), z))
            y = self.sigmoid(self.W2 @ z_b)
            y_pred.append(1 if y >= 0.5 else -1)
        return np.array(y_pred)


# =========================================================
# 4. CLASSE RBF
# =========================================================
class RBF:
    def __init__(self, X_train, y_train, n_centers=10, sigma=1.0):
        self.X = X_train
        self.y = y_train
        self.n_centers = n_centers
        self.sigma = sigma
        idx = np.random.choice(X_train.shape[1], n_centers, replace=False)
        self.centers = X_train[:, idx]

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
        self.w = np.linalg.pinv(phi_b.T) @ self.y
        erro = np.mean((self.y - phi_b.T @ self.w)**2)
        return [erro], 1

    def predict(self, X):
        phi = self._calc_phi(X)
        phi_b = np.vstack((-np.ones((1, phi.shape[1])), phi))
        y_pred = np.sign(phi_b.T @ self.w)
        return y_pred.flatten()


# =========================================================
# 5. FUNÇÕES AUXILIARES
# =========================================================
def split_balanceado(X, y, proporcao=0.8):
    X_pos = X[:, y == 1]
    X_neg = X[:, y == -1]
    Np, Nn = X_pos.shape[1], X_neg.shape[1]
    Np_train, Nn_train = int(Np * proporcao), int(Nn * proporcao)

    X_train = np.hstack((X_pos[:, :Np_train], X_neg[:, :Nn_train]))
    y_train = np.hstack((np.ones(Np_train), -np.ones(Nn_train)))
    X_test = np.hstack((X_pos[:, Np_train:], X_neg[:, Nn_train:]))
    y_test = np.hstack((np.ones(Np - Np_train), -np.ones(Nn - Nn_train)))

    idx_train = np.random.permutation(X_train.shape[1])
    idx_test = np.random.permutation(X_test.shape[1])
    return X_train[:, idx_train], y_train[idx_train], X_test[:, idx_test], y_test[idx_test]


def matriz_confusao_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[tn, fp], [fn, tp]])


def calcula_metricas(y_true, y_pred):
    cm = matriz_confusao_manual(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / (np.sum(cm) + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    sens = tp / (tp + fn + 1e-9)
    espec = tn / (tn + fp + 1e-9)
    f1 = 2 * (prec * sens) / (prec + sens + 1e-9)
    return acc, prec, sens, espec, f1


def plot_confusion(cm, titulo):
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(titulo)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()


def plot_curva_aprendizado(hist, titulo):
    if hist is None:
        print(f"[Aviso] Nenhum histórico para {titulo}")
        return
    hist = np.array(hist).reshape(-1)
    plt.figure()
    plt.plot(hist, 'o-', color='darkblue')
    plt.title(f'Curva de Aprendizado - {titulo}')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Médio')
    plt.grid(True)
    plt.show()


# =========================================================
# 6. VALIDAÇÃO MONTE CARLO (balanceado)
# ====================================================c=====
def monte_carlo(model_class, X, y, rodadas=10, **kwargs):
    metricas = {'acuracia': [], 'precisao': [], 'sensibilidade': [], 'especificidade': [], 'f1': []}
    melhores, piores = None, None
    melhor_acc, pior_acc = 0, 1
    hist_melhor, hist_pior = None, None

    for r in range(rodadas):
        if (r + 1) % 10 == 0 or r == 0:
            print(f"→ Rodada {r + 1}/{rodadas} concluída...")

        X_treino, y_treino, X_teste, y_teste = split_balanceado(X, y)

        modelo = model_class(X_treino, y_treino, **kwargs)
        hist, epocas = modelo.fit()
        y_pred = modelo.predict(X_teste)

        acc, prec, sens, espec, f1 = calcula_metricas(y_teste, y_pred)
        for nome, val in zip(metricas.keys(), [acc, prec, sens, espec, f1]):
            metricas[nome].append(val)

        if acc > melhor_acc:
            melhor_acc = acc
            melhores = (y_teste, y_pred)
            hist_melhor = hist
        if acc < pior_acc:
            pior_acc = acc
            piores = (y_teste, y_pred)
            hist_pior = hist

    print(f"✔ Finalizadas {rodadas} rodadas.\n")
    return metricas, melhores, piores, hist_melhor, hist_pior


# =========================================================
# 7. EXECUÇÃO
# =========================================================
data = np.loadtxt('spiral_d.csv', delimiter=',')
X = data[:, :2].T
y = np.where(data[:, 2] == 0, -1, 1)
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

plt.figure(figsize=(6,6))
plt.scatter(X[0, y==1], X[1, y==1], c='royalblue', label='Classe +1')
plt.scatter(X[0, y==-1], X[1, y==-1], c='crimson', label='Classe -1')
plt.legend(); plt.title("Distribuição das Classes"); plt.show()

modelos = [
    ("Perceptron", Perceptron, {'learning_rate':1e-3}),
    ("ADALINE", ADALINE, {'learning_rate':1e-3}),
    ("MLP", MLP, {'learning_rate':1e-3, 'hidden_neurons':10}),
    ("RBF", RBF, {'n_centers':10, 'sigma':1.0})
]

resultados = []
for nome, modelo, params in modelos:
    print(f"\nTreinando {nome}...")
    m, melhor, pior, hist_melhor, hist_pior = monte_carlo(modelo, X, y, **params)
    resultados.append((nome, m, melhor, pior, hist_melhor, hist_pior))

# =========================================================
# 8. RESULTADOS
# =========================================================
for met in ['acuracia', 'precisao', 'sensibilidade', 'especificidade', 'f1']:
    print(f"\n=== {met.capitalize()} ===")
    print(f"{'Modelo':<12}{'Média':>10}{'Desv.Pad':>12}{'Melhor':>12}{'Pior':>12}")
    for nome, m, *_ in resultados:
        media = np.mean(m[met])
        desvio = np.std(m[met])
        melhor = np.max(m[met])
        pior = np.min(m[met])
        print(f"{nome:<12}{media:>10.4f}{desvio:>12.4f}{melhor:>12.4f}{pior:>12.4f}")

# =========================================================
# 9. PLOTS
# =========================================================
for nome, _, melhor, pior, hist_melhor, hist_pior in resultados:
    plot_confusion(matriz_confusao_manual(*melhor), f"{nome} - Melhor Rodada")
    plot_confusion(matriz_confusao_manual(*pior), f"{nome} - Pior Rodada")
    plot_curva_aprendizado(hist_melhor, f"{nome} (Melhor)")
    plot_curva_aprendizado(hist_pior, f"{nome} (Pior)")

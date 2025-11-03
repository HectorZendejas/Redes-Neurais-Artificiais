 Redes Neurais Artificiais

## Introdução

Este trabalho é composto por duas etapas, nas quais serão utilizados modelos de Inteligência Artificial (IA) inspirados no funcionamento do cérebro humano. Esses modelos são as Redes Neurais Artificiais (RNA), tanto lineares quanto não lineares.

- **Etapa 1:** Solução de um problema simples, servindo como validador das implementações dos modelos solicitados.
- **Etapa 2:** Resolução de um problema de reconhecimento facial em imagens.

---

## Etapa 1: Regressão e Classificação para Problemas Bidimensionais (4,0 pts)

### Tarefa de Classificação

Utilize o conjunto de dados disponibilizado na plataforma AVA, chamado `spiral_d.csv`, que contém dados sintéticos com duas classes rotuladas na terceira coluna.

### Atividades

1. Organize o conjunto de dados para apresentação às redes neurais.
2. Realize uma visualização inicial dos dados por meio de gráfico de dispersão.
3. Implemente os seguintes modelos de RNA:
   - Perceptron Simples
   - ADALINE
   - Perceptron de Múltiplas Camadas (MLP)
   - Rede de Função de Base Radial (RBF)

   Para cada modelo, discuta a escolha dos hiperparâmetros.

4. Para os modelos MLP e RBF:
   - Identifique os casos de underfitting e overfitting.
   - Explore diferentes topologias modificando hiperparâmetros como número de neurônios e camadas.
   - Descreva as topologias encontradas.
   - Calcule os desempenhos utilizando:
     - Matriz de confusão
     - Acurácia
     - Especificidade
     - Sensibilidade
     - Curva de aprendizado

5. Valide os modelos com amostragem aleatória:
   - Realize R = 500 rodadas.
   - Em cada rodada, utilize 80% dos dados para treinamento e 20% para teste.
   - Métricas de desempenho: acurácia, sensibilidade, especificidade, precisão e F1-score.

6. Selecione os casos com maior e menor valor de cada métrica:
   - Construa a matriz de confusão (implementada pela equipe).
   - Utilize a biblioteca Seaborn para o plot (ex.: heatmap).
   - Plote a curva de aprendizado de todos os modelos.

7. Ao final das 500 rodadas, calcule para cada modelo:
   - Média aritmética
   - Desvio-padrão
   - Maior valor
   - Menor valor

   Apresente os resultados em tabelas (uma para cada métrica) e discuta os resultados. Os dados podem ser representados também por gráficos como Boxplot, Violin Plot, entre outros.

---

## Etapa 2: Classificação Multiclasse para Problema Multidimensional (6,0 pts)

### Reconhecimento Facial

Identifique 20 pessoas diferentes em imagens faciais. O conjunto de dados contém 640 imagens, cada uma com dimensão 120×128, disponível na pasta `RecFac` no AVA.

### Atividades

1. Escolha uma nova dimensão para redimensionar as imagens: (30×30), (40×40), (50×50) ou (80×80).
2. Organize a massa de dados **X** com dimensões \( \mathbb{R}^{p+1 \times N} \).
3. Codifique os rótulos com one-hot encoding. Exemplo para 5 pessoas:

   ```
   an2i   = [ +1, 1, 1, ..., 1 ]
   at33   = [ 1, +1, 1, ..., 1 ]
   boland = [ 1, 1, +1, ..., 1 ]
   bpm    = [ 1, 1, 1, +1, ..., 1 ]
   ch4f   = [ 1, 1, 1, 1, +1, ..., 1 ]
   ```

   Organize a massa de dados **Y** com dimensões \( \mathbb{R}^{C \times N} \).

4. Implemente os modelos:
   - Perceptron Simples
   - ADALINE
   - MLP
   - RBF

   Discuta os hiperparâmetros com base na experiência adquirida na Etapa 1.

5. Valide os modelos com simulações Monte Carlo:
   - Realize R = 10 rodadas.
   - Em cada rodada, utilize 80% dos dados para treinamento e 20% para teste.
   - Métrica de desempenho: acurácia.

6. Selecione os casos com maior e menor acurácia:
   - Construa as matrizes de confusão (implementadas pela equipe).
   - Utilize Seaborn para o plot.
   - Analise os erros e acertos por categoria.
   - Plote a curva de aprendizado de todos os modelos.

7. Ao final das 10 rodadas, calcule para cada modelo:
   - Média aritmética
   - Desvio-padrão
   - Maior valor
   - Menor valor

   Apresente os resultados em tabela e discuta-os. Utilize gráficos como Boxplot, Violin Plot, entre outros.

---

## Informações Adicionais sobre os Modelos de RNA

- Todos os dados devem ser normalizados (se ainda não estiverem).
- Todos os modelos devem possuir critério de convergência baseado no número máximo de épocas.
- O Perceptron Simples tradicional não possui esse critério, portanto deve ser modificado pela equipe.

### Funções de Ativação

- **Perceptron Simples e ADALINE:** degrau ou degrau bipolar.
- **MLP:** tangente hiperbólica ou sigmoide logística (usar a mesma função para todos os neurônios).

### Hiperparâmetros a considerar

1. Quantidade de camadas escondidas (MLP).
2. Quantidade de neurônios nas camadas escondidas (RBF possui apenas uma camada oculta).
3. Quantidade de neurônios na camada de saída.
4. Valor da precisão (critério de parada).
5. Número máximo de épocas.
6. Taxa de aprendizagem.

---

Se quiser, posso transformar esse conteúdo em um documento formatado ou ajudar com os gráficos e tabelas. Deseja seguir por esse caminho?

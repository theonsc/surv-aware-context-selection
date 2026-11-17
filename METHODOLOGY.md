# Metodologia e correspondências

## SurvPFN

O adaptador carrega a rede `SurvPFNNet`, a distribuição bar e as transformações da revisão upstream fixada. O caminho de inferência usa a API oficial; a otimização chama a rede PyTorch com os mesmos tensores de suporte e posição de separação. O `no_grad` da API pública é evitado apenas no caminho de treinamento. Parâmetros da rede ficam congelados e em modo de avaliação; o gradiente flui até covariáveis contínuas do contexto.

O contexto inicial fornece os tempos e indicadores de evento que ajustam `log1p` e a normalização por eventos. Esses valores e as categorias não mudam durante o treino. A consulta não fornece sua resposta à rede: sua sobrevivência entra somente na loss depois da inferência. Os pesos IPCW são calculados na amostra real de treino e permanecem fora do grafo.

## TuneTables clássico adaptado

| TuneTables para classificação | RiskSet-Tune v1 |
|---|---|
| PFN congelado como preditor | SurvPFN congelado |
| Prompt contínuo otimizado | Covariáveis contínuas X* otimizadas |
| Rótulo do prompt também inicializado/otimizado em variantes do original | Tempos e eventos das linhas selecionadas fixos |
| Cross-entropy ou KL segundo o protocolo TuneTables | NLL nativa censurada do SurvPFN, com IPCW para eventos |
| Consultas rotuladas durante ajuste | Consultas disjuntas do contexto; targets usados somente na loss |
| Contexto e inicialização avaliados no protocolo original | Mesmo suporte, seed, consultas, orçamento e split para TuneTables e RiskSet-Tune |

Assim, este baseline é uma adaptação no espaço de covariáveis, não uma reprodução integral das variantes experimentais de TuneTables. A métrica de validação para seleção do contexto é a loss preditiva básica sem regularização; Random não otimiza parâmetros.

## Hipótese RiskSet-Tune v1

Para evento i, seja `R(t_i) = {j: time_j >= t_i}`. Nas colunas contínuas, o contraste é `x_i - mean(x_j: j em R(t_i))`. Os eventos são agrupados em quantis temporais definidos nos dados de treino; as médias dos contrastes do contexto e da referência são comparadas após escala fixa por desvio padrão das covariáveis.

`L = L_native + lambda * L_risk`, com peso temporal proporcional à frequência de eventos na referência. Faixas sem eventos de contexto são omitidas e os pesos restantes renormalizados. A máscara de eventos e os tempos do contexto são fixos; portanto, a escolha das faixas não muda durante a otimização. λ=0 ignora por completo o termo de risco e segue o mesmo loop e objetivo do baseline.

Essa construção preserva apenas momentos médios simples dos conjuntos de risco. É uma hipótese modular, não uma decisão metodológica final. A escala de λ, número de faixas, sensibilidade a empates temporais, censura informativa e estabilidade entre seeds dependem de validação empírica.

## Avaliação

C-index IPCW e IBS usam a distribuição de censura estimada nos dados de treino reais. O horizonte máximo é determinado por quantil de treino e aplicado como censura administrativa para manter suporte. A grade IBS também vem de quantis de treino; se não estiver contida no acompanhamento do teste, IBS fica ausente com justificativa em vez de ajustar a grade ao teste. Comparações devem ser pareadas por dataset, split, K e seed.

## Piloto ampliado

`configs/extended_pilot.yaml` fixa seis datasets do SurvSet, K=32/64, seeds 42–44, λ={0,01; 0,1; 1} e marcos de 200, 1.000, 2.000 e 5.000 passos. As colunas categóricas, tempos e eventos permanecem fixos; atributos numéricos inteiros, como contagens, podem assumir valores fracionários durante a otimização e devem ser interpretados como uma relaxação contínua. O piloto registra o número de colunas otimizadas por contexto.

A loss e suas partes ficam separadas no histórico. A cada 100 passos registram-se normas e alinhamento entre gradientes da NLL e da penalidade ponderada. A seleção final entre lambdas usa apenas a menor NLL preditiva de validação; o relatório também mostra resultados por lambda para análise de sensibilidade. O Cox com L2 (`alpha=0,1`) é uma referência diagnóstica ajustada no treino real completo, com escala numérica e codificação categórica ajustadas somente nesse treino; seu orçamento difere do contexto sintético K.

O processo guarda checkpoints atômicos com parâmetros, Adam, estados dos geradores, melhor contexto, histórico e transformação temporal. `scripts/run_extended_pilot.py` percorre os marcos em ordem fixa e publica o relatório após três horas sem suspender o treino. A seleção dos próximos ajustes depende apenas do tempo e do progresso, nunca das métricas de teste. Os resultados de seeds com splits sobrepostos não representam replicações independentes.

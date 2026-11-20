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

## Variantes Score e Listwise

O experimento `configs/riskset_variants.yaml` trata os seis datasets já examinados como desenvolvimento. IBS é a métrica principal e a NLL preditiva de validação continua selecionando o contexto, inclusive o passo zero. O treino é pareado por dataset, seed e K; categorias, tempos, eventos e a transformação temporal são fixos.

**RiskSet-Score.** As covariáveis contínuas são padronizadas e categorias codificadas exclusivamente no treino real. Um Cox L2 com `alpha=0,1` fornece coeficientes fixos. Para cada sonda em `{0, 0,5β, β, 2β, ±e_j}` e cada evento, comparam-se suas covariáveis contínuas com a média do conjunto de risco ponderada por `exp(βᵀz)`. A penalidade é a média quadrática da diferença entre esses contrastes no contexto e no treino, agregada em até quatro faixas temporais com pesos de frequência dos eventos do treino. Faixas sem eventos no contexto são excluídas; a cobertura é registrada. O contraste com β=0 equivale ao objetivo anterior dividido pelo número de atributos contínuos, antes da média com outras sondas.

**RiskSet-Listwise.** Reutilizam-se os logits do batch preditivo. Para cada evento `i`, compara-se `r_j(T_i) = −log S_j(T_i)` no grupo que contém `i` e as consultas com `T_j > T_i`. A média dos termos `logsumexp(r_j) − r_i` é uma surrogate de ordenação no batch, não a verossimilhança parcial completa. Empates não são ordenados, censurados podem ser comparadores, e batches sem comparação válida têm loss zero. A sobrevivência diferenciável replica a convenção de barras e caudas da API oficial. A NLL mantém IPCW; o termo auxiliar não recebe outro peso de censura.

Cada objetivo auxiliar tem força fixa calibrada no contexto inicial: α é 0,1 vezes a mediana da razão entre as normas dos gradientes da NLL e da penalidade em cinco batches válidos, escolhidos por gerador separado em até 20 tentativas. Falhas de calibração são registradas. α=0 desvia diretamente para o treinamento de TuneTables. As escalas Score, Listwise e o λ=0,1 do RiskSet original não são diretamente comparáveis em unidades de loss.

Os checkpoints novos incluem configuração, sondas e estatísticas auxiliares, calibração, Adam, geradores, melhor contexto e a transformação temporal. O executor salva cada método ao terminar, calcula comparações somente em marcos concluídos e guarda prazo em `control.json`; editar atomicamente `deadline_epoch` permite mudar o prazo de um processo ativo. A extensão do prazo deve respeitar a autorização do usuário. O progresso e as estimativas usam somente tempos e jobs concluídos, não métricas de teste. Todos os achados continuam exploratórios.

## Baseline k-NN

Para cada consulta, `knn` seleciona K linhas do pool de referência por distância quadrática nas variáveis contínuas padronizadas e diferença categórica. Quando necessário, substitui os censurados mais distantes pelos eventos mais próximos para garantir três eventos no contexto. Cada consulta usa seu próprio contexto e sua própria transformação temporal. A seleção não usa desfechos da consulta.

# RiskSet-Tune

Infraestrutura experimental para comparar contextos Random, TuneTables adaptado e variantes de RiskSet-Tune com pesos congelados do SurvPFN. Os resultados de pilotos locais ficam sob `runs/` e são exploratórios.

## Estado

| Componente | Estado |
|---|---|
| Organização do pacote, tipos e configurações | Implementado |
| Splits e pré-processamento sem ajuste em teste | Implementado; requer validação com datasets reais |
| Inicialização e loop Adam compartilhado | Implementado; requer validação empírica |
| Regularização por contrastes de risco | Implementado como hipótese inicial, ainda sem validação empírica |
| Adaptador diferenciável SurvPFN | Verificado contra a API oficial e checkpoint local |
| Métricas IPCW/IBS e exportação | Implementado; pendente de validar grades em benchmarks |
| FinSurvival, RDS e PCA | Implementado como loader local inicial; pendente de validação no pacote oficial |
| Checkpoint retomável e diagnósticos de gradiente | Implementados; retomada comparada com trajetória contínua |
| Piloto ampliado em seis datasets, Cox e relatório parcial | Implementados; validação operacional curta concluída |

## Instalação

O SurvPFN e a versão fixada do TFM-Playground exigem Python 3.12. Instale Python 3.12 e `uv`, então:

```bash
uv sync --extra dev --extra survset --extra survpfn
```

Instale somente o módulo de rede do TFM-Playground na revisão usada pelos autores, sem puxar as dependências extras de treino deles:

```bash
uv pip install --no-deps 'tfmplayground @ git+https://github.com/automl/TFM-Playground.git@4afd33e1c60928c8425ac979b9e1fd092ad0f216'
```

Para FinSurvival, acrescente `--extra finsurvival`. O loader lê arquivos locais e não baixa dados. A dependência opcional de SurvSet pode consultar o catálogo quando o experimento selecionar `name: auto`.

Clone o código oficial na revisão que corresponde ao checkpoint:

```bash
git clone https://github.com/genepi-freiburg/SurvPFN.git /path/to/SurvPFN
git -C /path/to/SurvPFN checkout 4f26fe31efd811f546a6a3a12acd28293828ca7e
export SURVPFN_SOURCE=/path/to/SurvPFN
```

O caminho do checkpoint é sempre fornecido explicitamente. Baixe-o manualmente do repositório oficial de modelos, se necessário, e registre seu checksum junto aos artefatos.

## Executar futuramente

```bash
riskset-tune --config configs/pilot.yaml --checkpoint /path/to/survpfn_nr.pt --output runs/pilot
```

`configs/pilot.yaml` configura K=64; `configs/context_matrix.yaml` configura K=32, 64, 128 e 256. O comando executa os métodos e seeds declarados no YAML. Não há execução durante instalação/importação.

Para agregar um CSV de resultados já existente:

```bash
riskset-tune --output runs/pilot --list-results
```

## Piloto ampliado com retomada

`configs/extended_pilot.yaml` declara os seis datasets, K, seeds, lambdas, marcos e orçamento. A execução longa é instalada como serviço do usuário no macOS, com cache e fonte SurvPFN persistentes:

```bash
.venv/bin/python scripts/install_extended_launch_agent.py \
  --config configs/extended_pilot.yaml \
  --checkpoint artifacts/checkpoints/survpfn_nr.pt \
  --output runs/extended_pilot --start
```

O LaunchAgent usa `caffeinate` para evitar suspensão por inatividade. Mantenha o usuário conectado e o computador ligado. O estado fica em `runs/extended_pilot/status.json`; resultados por etapa, checkpoints de Adam, logs e o relatório parcial de três horas ficam no mesmo diretório. O treino retoma do último checkpoint de até 50 passos, e a execução para ao completar a matriz ou após o orçamento configurado.

Os resultados, configurações resolvidas, índices dos splits (`splits.npz`) e contextos são salvos no diretório de saída. Tabelas pareadas e um gráfico também são produzidos quando as dependências de visualização estão instaladas. Métricas sem suporte estatístico ficam ausentes com um motivo registrado; não se cria resultado fictício.

## Testes

```bash
pytest
SURVPFN_SOURCE=/path/to/SurvPFN SURVPFN_CHECKPOINT=/path/to/survpfn_nr.pt pytest -m integration
```

Os testes de integração são opcionais e requerem fonte oficial e checkpoint locais. A execução de um teste de contrato sem checkpoint não demonstra compatibilidade numérica com o modelo.

## Protocolo e limitações

Os grupos de inicialização, consultas, validação e teste são disjuntos. Imputação, mapeamento de categorias, PCA, pesos de censura e grade de avaliação usam somente dados de treino. Random seleciona linhas reais; TuneTables e RiskSet-Tune compartilham inicialização, consultas, orçamento e loss. Consulte [METHODOLOGY.md](METHODOLOGY.md) para a correspondência com TuneTables e a hipótese da regularização, e [NEXT_STEPS.md](NEXT_STEPS.md) para as validações pendentes.

## Comparação de variantes de conjuntos de risco

`configs/riskset_variants.yaml` configura RiskSet-Score e RiskSet-Listwise, além das referências anteriores. A definição dos objetivos, a calibração da força e as limitações do experimento estão em [METHODOLOGY.md](METHODOLOGY.md). O executor retomável é `python -m riskset_tune.experiments.variants_pilot`; `scripts/install_variants_launch_agent.py` prepara o LaunchAgent independente no macOS. Resultados são escritos em um diretório novo sob `runs/`.

# Reprodução dos experimentos

## Ambiente e insumos do benchmark

O benchmark roda em CPU com Python 3.12. O executor registra versões, hashes e a configuração em um novo manifesto no diretório da execução. A partir da raiz:

```sh
uv sync --locked --python 3.12 --extra dev --extra survset --extra survpfn --extra finsurvival
uv pip install --no-deps 'tfmplayground @ git+https://github.com/automl/TFM-Playground.git@4afd33e1c60928c8425ac979b9e1fd092ad0f216'

uv run --no-sync python -c "from pathlib import Path; Path('.local/upstream').mkdir(parents=True, exist_ok=True); Path('artifacts/checkpoints').mkdir(parents=True, exist_ok=True)"
git clone https://github.com/genepi-freiburg/SurvPFN.git .local/upstream/SurvPFN
git -C .local/upstream/SurvPFN checkout 4f26fe31efd811f546a6a3a12acd28293828ca7e
```

- **Checkpoint:** obtenha `survpfn_nr.pt` no [repositório oficial do SurvPFN](https://huggingface.co/samuelboehm/SurvPFN) e salve em `artifacts/checkpoints/`. O SHA-256 esperado é `103e6d59a845b056d9cb6620388e8fcd0e11216c6bae322d09e38c7d2ee4de2d`.
- **SurvSet:** o loader lê os datasets pelo pacote SurvSet, fixado em `uv.lock`.
- **FinSurvival:** obtenha os 24 arquivos RDS listados em [finsurvival-source.json](configs/finsurvival-source.json) do [repositório oficial na revisão usada](https://github.com/Large-Transaction-Models/DMLR_DeFi_Survival_Benchmark/tree/b0d487ffca4eedf445502a01d342d2470bb55426). Coloque-os em `.local/finsurvival/`, preservando o caminho a partir de `Survival_Data/` e retirando o prefixo `Data/`. Copie também `configs/finsurvival-source.json` para `.local/finsurvival/source.json`. O campo `fin_sha256` do manifesto registra os hashes dos RDS.

## Execução e conferência

O comando do README executa a matriz completa. Para validar a instalação em um dataset primeiro, use um diretório separado:

```sh
SURVPFN_SOURCE="$PWD/.local/upstream/SurvPFN" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m riskset_tune.experiments.full_benchmark \
  --config configs/full_benchmark_fixed_alpha.yaml --benchmark survset --dataset Aids2 \
  --checkpoint artifacts/checkpoints/survpfn_nr.pt --fin-root .local/finsurvival \
  --output runs/verificacao-aids2
```

Use apenas um processo por diretório de saída. Para retomar o benchmark, aguarde o processo anterior terminar e repita o comando com o mesmo diretório, a mesma versão do código e o mesmo ambiente. Use um diretório novo ao trocar de versão ou plataforma. A matriz completa produz 1.296 registros. Em plataformas diferentes, o arredondamento numérico pode alterar as métricas e os hashes das predições.

### Windows (PowerShell)

Após preparar o ambiente e os dados, chame o módulo Python diretamente a partir da raiz:

```powershell
$env:SURVPFN_SOURCE = (Resolve-Path .local/upstream/SurvPFN).Path
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& .\.venv\Scripts\python.exe -m riskset_tune.experiments.full_benchmark --config configs/full_benchmark_fixed_alpha.yaml --output runs/reproducao
```

Após a conclusão do benchmark, use o comando `uv run --python 3.12 --script` da próxima seção para gerar as tabelas e figuras.

## Tabelas e figuras a partir de métricas salvas

Após completar o benchmark, o comando do README gera as saídas dentro da pasta da execução. Para gravá-las em outro local, use `--output-dir`:

```sh
uv run --python 3.12 --script scripts/reproduce_report.py \
  --run-dir runs/reproducao --output-dir runs/reproducao/relatorio-alternativo
```

Se as dependências do script já estiverem instaladas, também é possível rodar `python scripts/reproduce_report.py --run-dir runs/reproducao` diretamente.

O gerador usa a etapa 0 para os baselines. Para os prompts, usa a etapa de 2.000 passos, que registra o resultado do prompt com menor NLL de validação até esse limite. A Tabela 1 agrega primeiro os seis datasets por seed e calcula a margem do intervalo t de 95% sobre as três médias. A Tabela 2 compara prompt tuning com 32 linhas ao melhor baseline com 128, por IBS médio nas três seeds. Os mapas calculam a variação em relação ao aleatório no mesmo dataset, seed e tamanho de contexto antes de tirar a média.

Os tamanhos dos datasets também vêm da execução. No SurvSet, o gerador conta as linhas dos splits de treino, validação e teste. No FinSurvival, lê os totais da auditoria feita após a limpeza e a junção por ID, antes de amostrar o teste. As tabelas geradas são `ibs-c-index-por-benchmark.csv` e `compressao-32-vs-128.csv`, ambas em `tabelas/`.

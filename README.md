# Compressão de contexto para o SurvPFN

Comparamos prompt tuning, com e sem ordenação de risco, a três baselines de seleção de linhas no SurvSet e no FinSurvival. Os resultados estão no [relatório](relatorio-riskset.pdf).

## Reproduzir os experimentos

Prepare o ambiente, os dados e o checkpoint seguindo o [guia de reprodução](REPRODUCIBILITY.md). Na raiz do repositório:

```sh
sh scripts/run_fixed_alpha_benchmark.sh runs/reproducao \
  && uv run --python 3.12 --script scripts/reproduce_report.py --run-dir runs/reproducao
```

O primeiro comando executa o benchmark e salva as métricas em `runs/reproducao/results.csv`. O segundo usa a saída dessa execução para gerar duas tabelas em CSV e quatro figuras em PNG em `runs/reproducao/relatorio/`. O PDF pode ser lido diretamente.

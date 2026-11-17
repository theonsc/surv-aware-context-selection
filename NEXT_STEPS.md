# Próximos passos antes do piloto

1. **Ambiente** — Preparar Python 3.12; a revisão de SurvPFN fixa TFM-Playground/PyTorch 2.9. O ambiente disponível durante a implementação contém Python 3.9/3.10 sem Torch, scikit-learn, pytest ou scikit-survival.
2. **Dependências e licenças** — Instalar extras em ambiente limpo; confirmar resolução da revisão TFM-Playground, `pfns` e pacotes SurvSet; guardar avisos MIT/Apache ao redistribuir componentes upstream.
3. **Contrato do modelo** — Com checkpoint local, verificar dimensões dos logits, igualdade do caminho diferenciável com `predict_logits`, previsões de sobrevivência, gradientes finitos em X* e `grad is None` em cada parâmetro congelado. Confirmar testes em CPU antes de tentar MPS.
4. **Loss** — Comparar o cálculo nativo com a implementação oficial em lote controlado: eventos, censuras, empates, extremos de tempo e pesos IPCW. A censura em tempo exato de evento e truncamento numérico precisam de conferência.
5. **Dados e protocolo** — Confirmar layout/catálogos do SurvSet na versão instalada, dataset escolhido, número de eventos por split e capacidade do checkpoint. Validar RDS, caminhos, IDs e seleção de atributos com um subconjunto local do FinSurvival; não usar o benchmark completo para testes de instalação.
6. **Sem vazamento** — Auditar alinhamento dos IDs e comprovar que consulta não aparece no suporte, que cada transformador é ajustado em treino e que nenhuma decisão de grade usa teste.
7. **Propriedades dos métodos** — Testar inicializações pareadas, λ=0 com trajetórias bit a bit idênticas sob mesmas seeds, persistência/recarregamento do contexto e melhor checkpoint pela loss de validação.
8. **Métricas** — Testar limites de suporte IPCW, grid IBS, conjuntos com poucos eventos e razões para métricas indefinidas. Revisar a decisão de censura administrativa no horizonte antes de relatar resultados.
9. **Piloto** — Só após as verificações anteriores, executar uma configuração K=64, seeds 42–44 e os três métodos. Registrar checkpoint/checksum, versões, splits e recursos. Resultados permanecem exploratórios até confirmar protocolo e sensibilidade dos hiperparâmetros.

Nenhuma validação empírica, download de checkpoint/dataset ou execução de piloto está incluída na entrega inicial.

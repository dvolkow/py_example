# Q1 (Методы обнаружения атак)

1. **Wavelet Analysis**. Идея -- наличие частотно-временной корреляции между легитимным трафиком и атакующим. Низкая вычислительная сложность (FWT ещё более быстрое, чем FFT -- O(n) вместо O(n logn)).
Метод годится для обнаружения атаки с довольно хорошей точностью, однако нужно подумать, как осуществить фильтрацию и принятие решений (возможно, нужно будет задействовать и другие методы вместе). Как *единственный* механизм в защите скорее всего не пригоден.
    * Li L, Lee G. DDoS attack detection and wavelets. Telecommunication Systems, 2005
    * Statistical Measures: Promising Features for Time Series Based DDoS Attack Detection, 2018. Можно обратить внимание, насколько отчетливо видны отличия обычного спектра от спектра трафика "под атакой" (с. 5)
2. **Support-vector data description (SVDD)**. Структурное прогнозирование и коллективная классификация пользователей.
    * Dick, U., & Scheffer, T. Learning to control a structured-prediction decoder for detection of HTTP-layer DDoS attackers. Machine Learning, 2016

3. **Entropy**. Обнаружение атак с помощью средств теории информации и статистического анализа. Метод основан на отличии распределений информационных метрик трафика для легитимного и вредоносного потоков.
    * Information Metrics for Low-rate DDoS Attack Detection : A Comparative Evaluation

La grafica de tokens procesados no esta bien porque muestra unos 80M tokens y son 475 millones, quizas no tiene en cuenta correctamente el gradient_accumulation

Nuestra implementacion de attention es purametne educativa, deberia tener un modulo de flash_attention mas "profesional", tomando como inspiracion a nanochat
# Training metrics:

* La grafica de tokens procesados no esta bien porque muestra unos 80M tokens y son 475 millones, quizas no tiene en cuenta correctamente el gradient_accumulation

# Training components

* Nuestra implementacion de attention es purametne educativa, deberia tener un modulo de flash_attention mas "profesional", tomando como inspiracion a nanochat

# Parallel disk writing for tokenizer

La PR que he hecho para tokenizacion + wirting en shards no me gusta, pero quizas hay algo intermedio donde escribimos un unico bloque pero podemos hacer cosas en paralelo y que no tarde tanto?
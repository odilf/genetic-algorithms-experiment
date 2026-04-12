#set text(lang: "es")
#show raw: set text(font: "IosevkaTerm NF")
#set page(numbering: "1.")

#align(horizon)[
    #text(size: 32pt)[Memoria proyecto IA de inspiración biológica] \
    Hecho por _Odysseas Machairas_ #text(fill:gray)[(100558761)]

    #v(2cm)

    #outline()
]

#pagebreak(weak: true)

#show link: set text(fill: blue.darken(00%))
#show link: underline

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#codly(languages: codly-languages, zebra-fill: white.darken(2%))

#set heading(numbering: "1.")

= Introducción

Esta es una breve memoria de las técnicas utilizadas y los resultados obtenidos para optimizar un conjunto de atributos utilizando algoritmos inspirados biológicamente.

He utilizado un algoritmo relativamente simple y me he centrado en los aspectos fundamentales del planteamiento del problema, con resultados satisfactorios en mi opinión. La implementación la he hecho en Python a mano, sin prácticamente uso de IA ni librerías aparte de `numpy` (y `matplotlib`), para poder entender a un nivel más granular cómo funcionan estos algoritmos.

Esta entrega también está disponible como Notebook en la entrega y:
- Como notebook (interactivo!) online: https://odilf.github.io/genetic-algorithms-experiment/
- Como notebook no interactivo: https://odilf.github.io/genetic-algorithms-experiment/static.html
- El código fuente está disponible en: https://github.com/odilf/genetic-algorithms-experiment.


= Descripción del problema

El problema se trata de, dados unos atributos de entrada y un atributo de salida de un conjunto de datos, encontrar una manera de optimizar los atributos para que una regresión haga una mejor predicción del atributo objetivo.
Entendemos con "optimización" o que la predicción linear tenga un error medio más pequeño (en nuestro caso utilizamos el MAE o _mean average error_) o que con menos atributos tengamos rendimiento similar o mejor. En mi caso, me he centrado más en reducir el error y no tanto en reducir el número de atributos.

Una de los principales pilares que me he basado en el proyecto es que, intuitivamente, parece que *el espacio de soluciones será caótico*. Por tanto es importante maximizar la exploración: no converger demasiado rápido para no quedarnos en mínimos locales, sino buscar mejores valores globales.

La manera que he modelado el problema es considerando que un conjunto de atributos es un "individuo". Esto lo he hecho porque así es muy fácil medir el fitness de un individuo de manera directa, global y objetiva. La alternativa sería considerar los atributos sintetizados como los individuos, pero entonces el fitness se mide en conjunto, no independientemente. Esto es problemático considerando que quiero maximizar la exploración, así que quiero poder comparar globalmente los individuos, no solo localmente en relación a otros.


```py
class Attribute:
    def __init__(self, op, data):
        self.op = op
        self.data = data

    def __repr__(self):
        return f"Attr {self.op}"

class Individual:
    def __init__(self, attrs, target):
        self.attrs = attrs
        self.target = target
        self.fitness = None

    def __repr__(self):
        return f"Individual {{ f={self.fitness}, attrs={self.attrs} }}"
```

= Descripción de la técnica

Mi algoritmo es un algoritmo génetico esencialmente basado en crear atributos como árboles de operaciones de los atributos originales (equivalente a expressiones-S, aunque en ningún momento guardo el árbol en sí). El algoritmo consiste en:

1. Crear una población inicial
2. Reproducir/crear descendencia de la población
3. Seleccionar los mejores individuos y purgar el resto
4. Volver a 2.

Esta es la idea clásica del algoritmo evolutivo. En el @fig-main-impl vemos en su completitud la implementación principal del algoritmo (llamando varias funciones, que son las que diseñaremos a continuación).

#figure(
    ```py
    def find_best_attributes(
        df,
        individual_size=20,
        pop_size=10,
        reproduction=10,
        select_round_size=4,
        mutation_rate=0.8,
    ) -> Iterator[Individual]:
        # Seleccionamos los attributos.
        initial_individual = as_individual(df)
        yield initial_individual

        population = generate_initial_population(
            initial_individual, individual_size=individual_size, pop_size=pop_size
        )

        while True:
            for _ in range(pop_size * (reproduction - 1)):
                i_a = floor(np.random.rand() * len(population))
                i_b = floor(np.random.rand() * len(population))

                new_individual = cross(population[i_a], population[i_b])
                if np.random.rand() <= mutation_rate:
                    mutate(new_individual)
                population.append(new_individual)

            population = select(
                population,
                pop_size=pop_size,
                round_size=select_round_size,
            )

            yield min(population, key=fitness)
    ```,
    caption: [Implementación principal del algoritmo genético.],
) <fig-main-impl>

== Población inicial

Es importante que haya variedad genética en cualquier algoritmo evolutivo, y en nuestro caso aún más porque queremos maximizar exploración. El problema es que fundamentalmente partimos de un individuo: el conjunto de atributos originales. Por ente, la variedad genética hay que generarla artificalmente.

Lo que hago es crear individuos compuestos de $N$ atributos ($N$ siendo parámetro que se pasa desde fuera), donde cada atributo está compuesto por un árbol aleatorio de operaciones de los atributos originales. El árbol se contruye recursivamente con una probabilidad de seguir creciendo que se reduce mientras crece, para tener árboles no demasiado grandes y pero no necesariamente simétricos.


```py
def generate_attr(
    individual: Individual, grow_prob=0.9, limit=20, ops=ops, seed=67
) -> Attribute:
    def generate_attr_impl(limit, grow_prob):
        if limit is 0 or np.random.rand() >= grow_prob:
            r = np.random.rand()
            return individual.attrs[
                np.floor(r * len(individual.attrs)).astype(np.int32)
            ]

        left = generate_attr_impl(limit - 1, grow_prob / 2)
        right = generate_attr_impl(limit - 1, grow_prob / 2)
        return cross_attr(left, right)

    return generate_attr_impl(limit, grow_prob)

def generate_initial_population(
    initial_individual: Individual, pop_size, individual_size
) -> list[Individual]:
    return [
        *[
            Individual(
                [
                    generate_attr(initial_individual)
                    for _ in range(individual_size)
                ],
                target=initial_individual.target,
            )
            for _ in range(pop_size)
        ],
        initial_individual,
    ]
```

== Reproducción: cruces y mutaciones

Los cruces son poco interesantes, simplemente hago cruces multipunto:

```py
def cross(a: Individual, b: Individual) -> Individual:
    # assert len(a.attrs) == len(b.attrs)
    a_len = floor(np.random.rand() * len(a.attrs))
    a_start = floor(np.random.rand() * len(a.attrs))

    if a_start + a_len >= len(a.attrs):
        a_start = (a_start + a_len) % len(a.attrs)
        a, b = b, a

    # assert a.target == b.target
    return Individual(
        [
            *b.attrs[:a_start],
            *a.attrs[a_start : a_start + a_len],
            *b.attrs[a_start + a_len :],
        ],
        a.target,
    )
```

Las mutaciones son más interesantes. Para mutar un individuo, selecciono dos de sus atributos y los cruzo con una operación binaria aleatoria. Esto hace que tengamos un atributo menos, así que ese lo regenero de manera aleatoria con la misma rutina que utilizo para generar la población inicial. Yo creo que esto es efectivo por dos razones. Por un lado, los nuevos atributos que se generan ayuda a que no se agote la variedad génetica. Esto es importante para la exploración que siempre decimos. Por el otro lado, el "cruce" de atributos se puede pensar como una mutación local, pero al final los resultados son bastante distintos. Este tipo de mutaciones son de las mejores, porque es una operación que tiene mucho sentido en el contexto del problema, y que aún así crea atributos nuevos significativamente distintos. Una manera de verlo es que en vez de moverse por el espacio de valores que solemos visualizar, se mueve por el espacio de operaciones que es más relevante en nuestro caso y ayuda a aislar los efectos importantes de los no tan importantes.

```py
def mutate(x: Individual) -> None:
    i_a = floor(np.random.rand() * len(x.attrs))
    i_b = floor(np.random.rand() * len(x.attrs))

    x.attrs[i_a] = cross_attr(x.attrs[i_a], x.attrs[i_b])
    x.attrs[i_b] = generate_attr(x, grow_prob=0.7)
```

== Selección de individuos

Finalmente, para la selección de individuos, como hemos dicho antes es fácil calcular el fitness haciendo la regresión lineal y calculando el MAE. En cuanto al algoritmo de selección, creo que sería un error coger los $n$ mejores individuos, ya que queremos que haya más exploración y no converger demasiado rápido. Por tanto, he elegido un algoritmo por tornos que mantiene una mayor variedad y esperamos que eventualmente esto encuentre mejor soluciones. 

```py
def select(population, pop_size, round_size=4):
    survivors = []
    pool = list(range(0, len(population)))
    for _ in range(pop_size):
        selected = [
            pool.pop(floor(np.random.rand() * len(pool)))
            for _ in range(round_size)
        ]

        best = None
        best_fit = float("inf")
        for i in selected:
            fit = fitness(population[i])
            if fit <= best_fit:
                best = population[i]
                best_fit = fit

        # assert best is not None
        survivors.append(best)

    return survivors
```

= Resultados

Vemos los resultados en la @fig-diabetes y @fig-california.

#figure(
    image("./figures/Diabetes.svg"),
    caption: [
        Resultados para el dataset de Diabetes.

        #align(left)[
            *Parámetros:*
            - generaciones=151
            - tamaño de población=295
            - número de atributos=30
        ]
    ]
) <fig-diabetes>

#figure(
    image("./figures/California.svg"),
    caption: [Resultados para el dataset de California
        #align(left)[
            *Parametros:*
            - generaciones=6
            - tamaño de población=50
            - número de atributos=20
        ]
    ]
) <fig-california>

Estos resultados no están nada mal! Considerando que es una implementación de puro Python del algoritmo en sí (aunque por supuesto la parte de cálculo numérico utiliza `numpy`) me parece bastante satisfactorio, en mi opinón, que solo ha tardado 90 segundos para un 13.63% de mejora en California y 6 minutos para un 24.55% de mejora Diabetes.

== Otras observaciones

He hecho algún experimento probando qué pasa si variamos algunos de los híperparametros del modelo.

En primer lugar, en la @fig-attrs vemos cuál es la mejora en relación al número de atributos. En teoría, podemos utilizar menos de $8$ atributos por individuo para intentar encontrar una mejor solución. En la práctica, los resultados empeoran hasta llegar a por lo menos 15 atributos. Estos tests no son del todo representativos ya que utilizo parámetros más pequeños que para los de la simulación inicial, pero eso no cambia que el método que he presentado no tiene pensado que haya tan pocos atributos.

Para que mejore el rendimiento, se podría hacer un proceso de selección de atributos _a posteriori_ de obtener los atributos con este método. Sin embargo, eso no es en lo que me centré para este trabajo.

#figure(
    image("./figures/attributes.svg"),
    caption: [Mejora en relación al número de atributos para el dataset Diabetes.]
) <fig-attrs>

En segundo lugar, la @fig-popsize enseña la relación de la mejora al tamaño de la población. Aquí vemos que claramente a mayor tamaño de población, mejor rinde el modelo. Esto es de esperar en general, pero aún más en nuestro caso que queremos maximizar la exploración. Con más individuos por generación, más se explora y más variedad genética hay, aunque por supuesto cada iteración tarda más. 

#figure(
    image("./figures/popsize.svg"),
    caption: [Mejora en relación al tamaño de la población para el dataset Diabetes.]
) <fig-popsize>

Finalmente, me parecía curioso ver qué pinta tienen los mejores atributos. Pues, para el set de californa, son estos:

- `Attr (MedInc * Longitude)`
- `Attr (AveOccup / MedInc)`
- `Attr (MedInc / AveOccup)`
- `Attr (Latitude + (MedInc * (Latitude + Latitude)))`
- `Attr (AveRooms - ((MedInc - HouseAge) / Latitude))`
- `Attr ((HouseAge + Latitude) / AveOccup)`
- `Attr HouseAge`
- `Attr ((HouseAge * MedInc) * Population)`
- `Attr (AveRooms * MedInc)`
- `Attr ((HouseAge * Longitude) + ((MedInc * Longitude) - (AveBedrms + Latitude)))`
- `Attr (AveOccup + (AveOccup + HouseAge))`
- `Attr (Longitude / AveBedrms)`
- `Attr (HouseAge * Longitude)`
- `Attr (((HouseAge / (AveRooms / AveBedrms)) + Latitude) - (Longitude / Latitude))`
- `Attr ((AveBedrms - HouseAge) * Latitude)`
- `Attr ((AveOccup + AveRooms) - Latitude)`
- `Attr ((AveBedrms * AveOccup) - Population)`
- `Attr ((AveBedrms + HouseAge) + MedInc)`
- `Attr (Longitude + AveOccup)`
- `Attr ((AveRooms - (Longitude + HouseAge)) * ((Latitude / AveRooms) * (Latitude / Latitude)))`

Algunos no tienen sentido aparentemente lógico, pero algunos otros sí, como `MedInc / AveOccup` (ingresos medios por persona) o simplemente `HouseAge`, que tiene sentido que predigan cuánto costará la casa.

= Conclusión

Un método simple evolutivo consigue sintetizar atributos nuevos con los cuales se puede hacer una mejor regresión linear. Centrándome en maximizar la exploración y con un buen algoritmo de mutación he conseguido resultados decentes relativamente rápido, incluso con una implementacion escrita principalmente en Python.

Eso es todo, gracias por leer :)

#pagebreak(weak: true)
#set heading(numbering: none)

= Anexo A. Uso de IA

Lo utilicé para escribir la celda con la regressión y el k-folds que utiliza `sklearn`. Todo el resto está escrito por mí.



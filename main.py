import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium", layout_file="layouts/main.slides.json")

with app.setup:
    import polars as pl
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    # Introducción

    He hecho la memoria integrada en el notebook que he utilizado para escribir el código. Creo que así será mas fácil de seguir. 

    - Se puede acceder al notebook (interactivo!) aquí: https://odilf.github.io/genetic-algorithms-experiment/
    - Si no funciona, hay una versión no interactiva aquí: https://odilf.github.io/genetic-algorithms-experiment/static.html
    - El código fuente está disponible en: https://github.com/odilf/genetic-algorithms-experiment.

    {mo.outline(label="Índice")}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Importar librerías y datos

    El primer paso es importar las librerías y los datos. No hay nada demasiado interesante aquí, excepto que estoy utilizando Polars en vez de Pandas ya que tiene alguna funcionalidad más (que probablemente no lleguemos a utilizar para este trabajo).
    """)
    return


@app.cell
def _():
    diabetes_df = pl.read_csv("./data/Diabetes.csv")
    mo.show_code(diabetes_df)
    return (diabetes_df,)


@app.cell
def _():
    california_df = pl.read_csv("./data/California.csv")
    mo.show_code(california_df)
    return (california_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Interpretación y codificación

    La primera gran pregunta es decidir qué es un individuo y qué es un gen. Mi primera inclinación es decir que un individio es un atributo, pero eso tiene el problema de que es díficil medir su fitness individualmente. Como al final hacemos una regresión lineal entre _todos_ los atributos, no está claro que porque un attributo tenga una mejor correlación lineal que todos los attributos juntos la tengan.

    Por tanto, yo creo que rendirá mejor interpretar un individuo como un conjunto de atributos. Esto hace que los attributos sean los genes del individuo.

    Hagamos clases para los dos para tener el codigo organizadito.
    """)
    return


@app.cell(hide_code=True)
def _(california_df, diabetes_df):
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


    def as_individual(df) -> Individual:
        """Transforma un dataframe a una lista de attributos"""
        input = df.columns[:-1]
        target = df.columns[-1]

        return Individual(
            [
                Attribute(name, series.to_numpy())
                for (name, series) in zip(input, df)
            ],
            df[target].to_numpy(),
        )


    diabetes = as_individual(diabetes_df)
    california = as_individual(california_df)

    mo.show_code()
    return Attribute, Individual, as_individual, california, diabetes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cruces de attributos

    Es fácil pensar cómo cruzar atributos. Vamos a tener una lista de operadores binarios y para generar un nuevo atributos elegimos un operador y lo aplicamos a los atributos.

    Dado que los atributos son genes, los "cruces de atributos" son una manera de generar mutaciones de genes. En la sección de mutaciones comento más esto.
    """)
    return


@app.cell(hide_code=True)
def _():
    numeric_ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        (
            "/",
            # Safe division
            lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0),
        ),
    ]

    logical_ops = [
        # ("&", lambda a, b: a.astype(np.int32) & b.astype(np.int32)),
        # ("|", lambda a, b: a.astype(np.int32) | b.astype(np.int32)),
        # ("^", lambda a, b: a.astype(np.int32) ^ b.astype(np.int32)),
    ]

    ops = [*numeric_ops, *logical_ops]

    mo.show_code()
    return (ops,)


@app.cell(hide_code=True)
def _(Attribute, ops):
    def cross_attr(a: Attribute, b: Attribute, ops=ops) -> Attribute:
        (op_sym, op) = ops[np.floor(np.random.rand() * len(ops)).astype(np.int32)]

        return Attribute(f"({a.op} {op_sym} {b.op})", op(a.data, b.data))


    mo.show_code()
    return (cross_attr,)


@app.cell(hide_code=True)
def _(example_attr):
    mo.md(rf"""
    ## Individuos iniciales

    Ahora podemos pensar cómo inicializamos la población y cómo generamos descendencias de poblaciones.

    Para la inicialización, tiene sentido empezar con los atributos originales, pero por supuesto necesitamos más individuos con la mayor variedad genética posible. Para hacer esto, cogemos los atributos originales y empezaremos a hacer combinaciones aleatorias. Vamos a utilizar una probabilidad $p$ de que crezca el árbol de operaciones y tener un límite de tamaño. Para que haya variedad de tamaños sin que se hagan demasiado grandes los árboles, vamos a disminuir la probabilidad de que crezca el árbol cada vez que decide crecer.

    Puede sonar un poco preocupante el hecho que empezamos con un solo individuo, pero espero que estas mutaciones por doquier proporcionen suficiente variedad genética.

    Un ejemplo de un attributo generado es el siguiente: `{example_attr}`
    """)
    return


@app.cell
def _(Attribute, Individual, cross_attr, ops):
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

    mo.show_code()
    return (generate_attr,)


@app.cell
def _(diabetes, generate_attr):
    example_attr = generate_attr(
        diabetes,
        grow_prob=0.9,
    )
    return (example_attr,)


@app.cell(hide_code=True)
def _(Individual, diabetes, generate_attr):
    def generate_initial_population(
        initial_individual: Individual, pop_size, individual_size
    ) -> list[Individual]:
        return [
            Individual(
                [
                    generate_attr(initial_individual)
                    for _ in range(individual_size)
                ],
                target=initial_individual.target,
            )
            for _ in range(pop_size)
        ]


    mo.show_code(generate_initial_population(diabetes, pop_size=10, individual_size=3))
    return (generate_initial_population,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Mutaciones y cruces

    Con esto podemos hacer funciones de cruzar y mutar genes de individuos.

    ## Cruces

    Para el cruce no nos vamos a complicar y simplemente vamos a coger un cruce multipunto. No hay gran justificación para esta decisión más alla que no me quería complicar demasiado la vida.
    """)
    return


@app.function
def floor(x):
    return np.floor(x).astype(np.int32)


@app.cell
def _(Individual):
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


    mo.show_code()
    return (cross,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mutaciones

    Como he comentado antes, el cruce de atributos es un "cruce de genes" que al final es una especie de mutación. Mi rutina de mutación selecciona dos genes/atributos y los cruza con la rutina anterior. Sin embargo, esto genera un atributo pero hemos utilizado dos para sacarlo. Uno de los genes utilizados lo reemplazo con el nuevo, pero aún quedaría uno antiguo. Yo creía que sería malo dejarlo porque perderíamos variedad genética (que es precisamente lo que tratamos de evitar con las mutaciones), así que el otro gen lo reemplazo con un gen nuevo, utilizando la rutina `generate_attr` comentada anteriormente.

    Yo creo que es buena idea hacerlo conde esta manera porque espero que así el individuo se quede buscando mas en su región del espacio de posibilidades, mientras que una mutación totalmente aleatoria temo que le cueste converger a buenas soluciones; pero también tiene una parte aleatoria para no perder variedad genética.
    """)
    return


@app.cell(hide_code=True)
def _(Individual, cross_attr, generate_attr):
    def mutate(x: Individual) -> None:
        i_a = floor(np.random.rand() * len(x.attrs))
        i_b = floor(np.random.rand() * len(x.attrs))

        x.attrs[i_a] = cross_attr(x.attrs[i_a], x.attrs[i_b])
        x.attrs[i_b] = generate_attr(x, grow_prob=0.7)


    mo.show_code()
    return (mutate,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Algoritmo genético
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Selección de individuos

    Tenemos que escribir una función que haga la selección, que es la parte que imita a la selección natural. Quiero que haya bastante exploración, así que he cogido un formato de torneo para no quedarme solo con los mejores y no converger a un minímo local demasiado rápido.
    """)
    return


@app.cell(hide_code=True)
def _(fitness):
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


    mo.show_code()
    return (select,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cálculo de error (fitness)

    Estoy dividiendo entre datos de entrenamiento y validación, y utilizo k-folds porque en general tengo entendido que es un método más robusto de calcular el error.
    """)
    return


@app.cell(hide_code=True)
def _():
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.linear_model import LinearRegression

    def error(Y, y):
        return np.mean(np.abs(Y - y), axis=0)


    def mean_error(Y, y):
        return np.mean(error(Y, y))


    def fitness(individual, k_cross=5):
        if individual.fitness is not None:
            return individual.fitness

        X = np.array([attr.data for attr in individual.attrs]).T
        y = individual.target

        kf = KFold(n_splits=k_cross, shuffle=True, random_state=69420)
        errors = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = regress(X_train, y_train)
            y_pred = model.predict(X_test)
            errors.append(mean_error(y_test, y_pred))

        fitness = np.mean(errors)
        individual.fitness = fitness
        return fitness

    def regress(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model


    mo.show_code()
    return (fitness,)


@app.cell(hide_code=True)
def _(california, diabetes, fitness):
    mo.md(f"""
    - Fitness inicial para diabetes: {fitness(diabetes):.3f}
    - Fitness inicial para california: {fitness(california):.4f}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## El algoritmo genético en sí.

    Al fin, podemos escribir el algoritmo génetico en sí.
    """)
    return


@app.cell
def _(
    Individual,
    as_individual,
    cross,
    fitness,
    generate_initial_population,
    mutate,
    select,
):
    from typing import Iterator


    def find_best_attributes(
        df,
        individual_size=20,
        pop_size=10,
        reproduction=10,
        select_round_size=4,
        mutation_rate=0.1,
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


    mo.show_code()
    return (find_best_attributes,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Evaluación y conclusiones
    """)
    return


@app.function
# Movidas para que los plots sean reactivos
def slice_gen_as_array(gen, title=None):
    results = []

    def index(i):
        if i < 0:
            return results[i]

        r = range(len(results), i + 1)
        # Show nice progress bar if there are enough generations
        if i - len(results) >= 10:
            r = mo.status.progress_bar(
                r,
                title=f"Generando attributos para {title}",
                remove_on_exit=True,
            )

        for i in r:
            results.append(next(gen))

        return results[:i]

    return index


@app.cell
def _(diabetes_df, find_best_attributes, popsize_slider):
    results_diabetes = slice_gen_as_array(
        find_best_attributes(diabetes_df, pop_size=popsize_slider.value),
        "Diabetes",
    )
    return (results_diabetes,)


@app.cell
def _(california_df, find_best_attributes):
    results_california = slice_gen_as_array(
        find_best_attributes(california_df, pop_size=20),
        "California",
    )
    return (results_california,)


@app.cell
def _(fitness):
    def plot_results(results, generations=50, title=None):
        individuals = results(generations)
        f_start = fitness(individuals[0])
        f_end = fitness(individuals[-1])
        improvement = (f_start - f_end) / f_start

        plt.figure(dpi=200, figsize=(12, 6))
        plt.title(f"{title} (mejora={improvement * 100:.3f}%)")
        plt.plot([fitness(ind) for ind in individuals])
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.grid(alpha=0.2)
        return mo.vstack(
            [
                mo.stat(
                    value=f"+{improvement * 100:.3f}%",
                    label="Mejora",
                    direction="increase",
                    caption=f"De {f_start:.2f} a {f_end:.2f}",
                ),
                plt.gcf(),
            ]
        )
    return (plot_results,)


@app.cell(hide_code=True)
def _():
    generations_slider = mo.ui.slider(10, 500, full_width=True, show_value=True, debounce=True, value=30)
    popsize_slider = mo.ui.slider(10, 500, full_width=True, show_value=True, debounce=True, value=150)
    mo.vstack(
        [
            mo.md("Número de generaciones"),
            generations_slider,
            mo.md("Tamaño de la población"),
            popsize_slider,
        ]
    )
    return generations_slider, popsize_slider


@app.cell
def _(generations_slider, plot_results, results_diabetes):
    plot_results(
        results_diabetes,
        generations=generations_slider.value,
        title="Diabetes",
    )
    return


@app.cell
def _(plot_results, results_california):
    plot_results(
        results_california,
        generations=15,
        title="California",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Efecto del número de atributos (i.e., genes)

    Esperaríamos que cuantos más atributos mejor el modelo porque además la regresión linear tiene libertad de ignorar atributos si son malos. Y en general vemos que en efecto mejora, pero no mucho.

    A veces para $n=1$ el modelo rinde mejor que para $n=2$ y $n=3$, pero esto no es consistente y asumo que es sobre todo anecdótico. En general, si no tenemos suficientes atributos empeora el rendimiento porque la primera generación ya es muy mala. Se podría intentar resolver esto, pero no lo he hecho.
    """)
    return


@app.cell
def _(diabetes_df, find_best_attributes, fitness):
    def _():
        number_of_attributes = [1, 2, 3, 5, 10, 15, 20, 30]

        plt.figure(dpi=200, figsize=(12, 6))
        for n in number_of_attributes:
            results = find_best_attributes(diabetes_df, individual_size=n)
            individuals = [
                next(results)
                for _ in mo.status.progress_bar(
                    range(10), remove_on_exit=True, title=f"Computing {n=}"
                )
            ]

            f_start = fitness(individuals[0])
            f_end = fitness(individuals[-1])
            improvement = (f_start - f_end) / f_start

            plt.plot([fitness(ind) for ind in individuals], label=f"{n=}, mejora={improvement*100:.2f}%")
        plt.legend()
        return plt.gcf()


    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Efecto del tamaño de la población

    En la figura abajo se ve como rinde el algoritmo con tamaños distintos de población. Este es un caso donde se ve muy claramente que a mayor población, mejor rendimiento. El problema fundamental es que aumentar el tamaño de la población hace que el algoritmo corra mucho más lento.

    Otra cosa curiosa es que parece que la aproximación puede seguir mejorando para poblaciones más grandes.
    """)
    return


@app.cell
def _(diabetes_df, find_best_attributes, fitness):
    population_sizes = [5, 10, 20, 50, 100]

    plt.figure(dpi=200, figsize=(12, 6))
    for pop_size in population_sizes:
        _results = find_best_attributes(diabetes_df, pop_size=pop_size)
        _individuals = [
            next(_results)
            for _ in mo.status.progress_bar(
                range(30), remove_on_exit=True, title=f"Computing {pop_size=}"
            )
        ]

        f_start = fitness(_individuals[0])
        f_end = fitness(_individuals[-1])
        improvement = (f_start - f_end) / f_start

        plt.plot([fitness(ind) for ind in _individuals], label=f"{pop_size=}, mejora={improvement*100:.2f}%")
    plt.legend()
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(best_attrs):
    mo.md(f"""
    ## Los mejores atributos

    Y, al final, qué pinta tienen los mejores atributos? Pues, para el set de californa, son estos:
    ™
    - {"\n- ".join([f"{x}" for x in best_attrs])}
    """)
    return


@app.cell
def _(fitness, results_california):
    best_attrs = min(results_california(15), key=fitness).attrs
    return (best_attrs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Y eso es todo, muchas gracias por leer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Anexo

    ## Uso de IA

    Lo utilicé para la celda con la regressión y el k-folds que utiliza `sklearn`.
    """)
    return


if __name__ == "__main__":
    app.run()

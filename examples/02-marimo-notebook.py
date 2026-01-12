import marimo

__generated_with = "0.19.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example of marimo notebook with Landweber unfolding method
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from bssunfold import Detector
    return Detector, go, mo, pd


@app.cell
def _(Detector, pd):
    # response functions default GSF
    from bssunfold import RF_GSF
    df = pd.DataFrame.from_dict(RF_GSF, orient='columns')
    detector = Detector(df)
    return (detector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Input detector readings
    """)
    return


@app.cell
def _(mo):
    # Показания по умолчанию
    default_readings = {
        "0in": 3,
        "2in": 9,
        "3in": 36,
        "5in": 41,
        "6in": 219,
        "8in": 220,
        "10in": 172,
        "12in": 120,
        "15in": 66,
        "18in": 34,
    }

    # input
    reading_inputs = {}
    for sphere, value in default_readings.items():
        reading_inputs[sphere] = mo.ui.number(
            value=value,
            label=f"{sphere}",
            step=1,
            start=0.0
        )

    mo.hstack(
        [
            mo.vstack([reading_inputs["0in"], reading_inputs["2in"], reading_inputs["3in"]]),
            mo.vstack([reading_inputs["5in"], reading_inputs["6in"], reading_inputs["8in"]]),
            mo.vstack([reading_inputs["10in"], reading_inputs["12in"], reading_inputs["15in"]]),
            mo.vstack([reading_inputs["18in"]])
        ]
    )
    return (reading_inputs,)


@app.cell
def _(mo):
    # Выбор числа итераций
    iterations = mo.ui.slider(
        value=3000,
        start=100,
        stop=10000,
        step=100,
        label="Number of iterations"
    )

    iterations
    return (iterations,)


@app.cell(hide_code=True)
def _(detector, iterations, reading_inputs):
    # get the current data
    current_readings = {sphere: input.value for sphere, input in reading_inputs.items()}

    result = detector.unfold_landweber(
        current_readings, 
        max_iterations=iterations.value
    )
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Unfolded spectrum
    """)
    return


@app.cell
def _(go, result):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result["energy"],
        y=result["spectrum"],
        mode="lines",
        name="unfolded spectrum",
        line=dict(color="blue", width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 0, 255, 0.1)"
    ))

    fig.update_layout(
        xaxis_title="Energy, MeV",
        yaxis_title="Fluence per unit lethargy, F(E)E, neutron/(cm² ∙ s)",
        height=500,
        template="plotly_white",
        showlegend=True
    )

    fig.update_xaxes(type="log")

    fig
    return


if __name__ == "__main__":
    app.run()

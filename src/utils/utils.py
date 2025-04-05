import plotly.express as px
import plotly.graph_objects as go
def create_interactive_table(data, title, max_rows=None):
  
    if max_rows is not None:
        df = data.head(max_rows)
    else:
        df = data

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left'
        )
    )])
    fig.update_layout(title=title)
    return fig
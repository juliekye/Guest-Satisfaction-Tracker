import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pickle
from collections import defaultdict
from models.common import AugmentedReview, AugmentedTopic, SentimentType

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

POS_COLORS = ["rgba(0,100,0,0.7)", "rgba(0,128,0,0.7)", "rgba(0,160,0,0.7)",
              "rgba(0,190,0,0.7)", "rgba(0,220,0,0.7)"]
NEG_COLORS = ["rgba(139,0,0,0.7)", "rgba(178,34,34,0.7)", "rgba(205,92,92,0.7)",
              "rgba(220,20,60,0.7)", "rgba(255,69,0,0.7)"]

def preprocess_data(property_code: str):
    with open(f'data/{property_code}/reviews.pkl', 'rb') as f:
        reviews = pickle.load(f)
    with open(f'data/{property_code}/topics.pkl', 'rb') as f:
        topics = pickle.load(f)

    valid_pos_topics = {t.topic for t in topics if t.sentiment_type == SentimentType.POSITIVE}
    valid_neg_topics = {t.topic for t in topics if t.sentiment_type == SentimentType.NEGATIVE}

    p_line, n_line = [0]*12, [0]*12
    pos_data = [defaultdict(int) for _ in range(12)]
    neg_data = [defaultdict(int) for _ in range(12)]
    all_reviews = []

    for review in reviews:
        idx = review.review_date.month - 1
        sentiment = review.sentiment_type
        if sentiment == SentimentType.POSITIVE:
            p_line[idx] += 1
            for t in review.assigned_topics:
                if t in valid_pos_topics:
                    pos_data[idx][t] += 1
        elif sentiment == SentimentType.NEGATIVE:
            n_line[idx] += 1
            for t in review.assigned_topics:
                if t in valid_neg_topics:
                    neg_data[idx][t] += 1

        all_reviews.append({
            'month_idx': idx,
            'sentiment': sentiment,
            'topics': review.assigned_topics,
            'body': review.review.body,
            'rating': int(review.review.stars),
            'date': review.review_date.strftime('%Y-%m-%d')
        })

    return p_line, n_line, pos_data, neg_data, valid_pos_topics, valid_neg_topics, all_reviews

def create_main_figure(p_line, n_line, pos_data, neg_data, valid_pos_topics, valid_neg_topics):
    fig = go.Figure()

    # Plot top 5 positive/negative topics each month as stacked bars
    for rank in range(5):
        pos_y, pos_labels = [], []
        neg_y, neg_labels = [], []
        for i in range(12):
            pos_items = sorted(
                [(k, v) for k, v in pos_data[i].items() if k in valid_pos_topics],
                key=lambda x: -x[1]
            )[:5]
            neg_items = sorted(
                [(k, v) for k, v in neg_data[i].items() if k in valid_neg_topics],
                key=lambda x: -x[1]
            )[:5]

            if rank < len(pos_items):
                pos_y.append(pos_items[rank][1])
                pos_labels.append(pos_items[rank][0])
            else:
                pos_y.append(0)
                pos_labels.append('')
            if rank < len(neg_items):
                neg_y.append(neg_items[rank][1])
                neg_labels.append(neg_items[rank][0])
            else:
                neg_y.append(0)
                neg_labels.append('')

        fig.add_trace(go.Bar(
            x=MONTH_NAMES,
            y=pos_y,
            name=f"Pos Topic {rank+1}",
            marker_color=POS_COLORS[rank],
            offsetgroup='1',
            customdata=pos_labels,
            hovertemplate='%{customdata}<extra>Positive</extra>',
            showlegend=False
        ))

        fig.add_trace(go.Bar(
            x=MONTH_NAMES,
            y=neg_y,
            name=f"Neg Topic {rank+1}",
            marker_color=NEG_COLORS[rank],
            offsetgroup='2',
            customdata=neg_labels,
            hovertemplate='%{customdata}<extra>Negative</extra>',
            showlegend=False
        ))

    # Plot positive/negative lines on the secondary y-axis
    fig.add_trace(go.Scatter(
        x=MONTH_NAMES, y=p_line,
        name='Positive Reviews',
        mode='lines+markers',
        line=dict(color='green', width=4),
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=MONTH_NAMES, y=n_line,
        name='Negative Reviews',
        mode='lines+markers',
        line=dict(color='red', width=4),
        yaxis='y2'
    ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Topic Counts'),
        yaxis2=dict(overlaying='y', side='right', title='Total Reviews'),
        height=750,
        width=1400,
        margin=dict(l=40, r=40, t=40, b=60),
        plot_bgcolor='white'
    )
    return fig

def review_box(reviews, topic, month_name):
    header = html.H4(
        ["Topic: ", html.B(topic)],
        style={'marginBottom': '10px', 'color': '#333', 'fontWeight': 'bold'}
    )
    return html.Div([
        header,
        html.Div([
            html.Div([
                html.Div(r['body'], style={'fontSize': '15px', 'lineHeight': '1.6', 'marginBottom': '8px'}),
                html.Div('⭐' * r['rating'] + f" · {r['date']}", style={'fontSize': '12px', 'color': '#666'})
            ], style={
                'border': '1px solid #ddd',
                'borderRadius': '8px',
                'padding': '12px',
                'marginBottom': '12px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
                'backgroundColor': 'white'
            }) for r in reviews
        ], style={
            'maxHeight': '450px',
            'overflowY': 'auto',
            'border': '1px solid #ccc',
            'padding': '10px',
            'borderRadius': '8px',
            'backgroundColor': '#fff'
        })
    ])

app = dash.Dash(__name__)
app.title = "Guest Review Dashboard"

app.layout = html.Div(
    style={
        # Bright, obvious background gradient so you can see it:
        'background': 'linear-gradient(to right, #ffafbd, #ffc3a0)',
        'minHeight': '100vh',
        'padding': '50px',
        'boxSizing': 'border-box',
    },
    children=[
        html.Div(
            style={
                # Big, bright outline so you can see it:
                'border': '6px solid #ff6f91',
                'borderRadius': '12px',
                # Very pronounced box shadow:
                'boxShadow': '0 10px 20px rgba(0,0,0,0.2)',
                'backgroundColor': '#ffffff',
                'maxWidth': '1400px',
                'margin': '0 auto',
                'padding': '30px'
            },
            children=[
                html.H1(
                    "Guest Satisfaction Tracker",
                    style={
                        'textAlign': 'center',
                        'fontSize': '40px',
                        'marginBottom': '30px',
                        'color': '#333',
                        'fontWeight': 'bold'
                    }
                ),
                dcc.Dropdown(
                    id='property-dropdown',
                    options=[
                        {'label': 'Cambria Hotel Downtown Phoenix', 'value': 'az474'},
                        {'label': 'Mayfair Hotel, NY', 'value': 'ny900'},
                        {'label': 'Quality Inn Long Beach', 'value': 'ca460'}
                    ],
                    value='ny900',
                    clearable=False,
                    # Revert to the original minimal styling that shows the default arrow
                    style={
                        'width': '400px',
                        'margin': '0 auto 30px'
                    }
                ),
                html.Div(
                    dcc.Graph(id='main-chart'),
                    style={
                        'display': 'flex',
                        'justifyContent': 'center',
                        'marginBottom': '40px'
                    }
                ),
                html.Div(
                    style={
                        'marginTop': '20px',
                        'padding': '20px',
                        'border': '2px solid #ddd',
                        'borderRadius': '8px',
                        'boxShadow': '0 5px 10px rgba(0,0,0,0.1)',
                        'backgroundColor': 'white'
                    },
                    children=[
                        html.H3(
                            "Topic Breakdown",
                            style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#333', 'fontSize': '24px'}
                        ),
                        html.Div([
                            dcc.Graph(id='topic-pie', style={'width': '500px', 'flexShrink': 0}),
                            html.Div(id='review-box', style={'flexGrow': 1, 'marginLeft': '20px'})
                        ], style={
                            'display': 'flex',
                            'justifyContent': 'center',
                            'gap': '40px',
                            'alignItems': 'flex-start'
                        })
                    ]
                )
            ]
        )
    ]
)

@app.callback(
    Output('main-chart', 'figure'),
    Input('property-dropdown', 'value')
)
def update_main_chart(property_code):
    p_line, n_line, pos_data, neg_data, valid_pos_topics, valid_neg_topics, _ = preprocess_data(property_code)
    return create_main_figure(p_line, n_line, pos_data, neg_data, valid_pos_topics, valid_neg_topics)
@app.callback(
    [Output('topic-pie', 'figure'), Output('review-box', 'children')],
    [
        Input('main-chart', 'clickData'),
        Input('topic-pie', 'clickData'),
        Input('property-dropdown', 'value')
    ]
)
def update_pie_and_reviews(main_click, pie_click, property_code):
    p_line, n_line, pos_data, neg_data, valid_pos_topics, valid_neg_topics, all_reviews = preprocess_data(property_code)
    main_fig = create_main_figure(p_line, n_line, pos_data, neg_data, valid_pos_topics, valid_neg_topics)

    # Determine clicked bar sentiment + month
    if main_click:
        clicked_month = main_click['points'][0]['x']
        curve_number = main_click['points'][0]['curveNumber']
        trace_name = main_fig['data'][curve_number]['name']
        sentiment = 'positive' if 'Pos' in trace_name else 'negative'
    else:
        clicked_month = 'Jan'
        sentiment = 'positive'

    month_idx = MONTH_NAMES.index(clicked_month)
    if sentiment == 'positive':
        data = pos_data[month_idx]
        valid_topics = valid_pos_topics
        sentiment_type = SentimentType.POSITIVE
    else:
        data = neg_data[month_idx]
        valid_topics = valid_neg_topics
        sentiment_type = SentimentType.NEGATIVE

    # Build pie data
    items = [(k, v) for k, v in data.items() if k in valid_topics]
    items.sort(key=lambda x: -x[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    # Check if the old pie_click topic is still valid for the new property code
    if pie_click:
        candidate_topic = pie_click['points'][0]['label']
        if candidate_topic in labels:
            selected_topic = candidate_topic
        else:
            # If not valid, reset to first label or None
            selected_topic = labels[0] if labels else None
    else:
        selected_topic = labels[0] if labels else None

    # Filter reviews
    filtered = [
        r for r in all_reviews
        if r['month_idx'] == month_idx 
        and r['sentiment'] == sentiment_type 
        and selected_topic in r['topics']
    ]

    # Create pie
    pie = go.Figure(go.Pie(labels=labels, values=values))
    pie.update_layout(title=f"{sentiment.capitalize()} Topics – {clicked_month}", height=450)

    return pie, review_box(filtered, selected_topic, clicked_month)


if __name__ == '__main__':
    app.run(debug=True)

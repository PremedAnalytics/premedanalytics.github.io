from enum import unique
import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx
import plotly.graph_objects as go

df = pd.read_csv("sdnThreads.csv")
df = df.loc[:, ('school', 'ids')]

# Create School List Key/Dictionary
schoolList = pd.Series.unique(df['school'])
schoolKey = pd.DataFrame(
    {'schoolId': range(1, len(schoolList)+1), 'schoolList': schoolList})
allSchools = pd.unique(schoolKey['schoolId'])
schoolRef = schoolKey.set_index('schoolId').to_dict()


userActive = df.merge(schoolKey, how="left",
                      left_on='school', right_on='schoolList')
userActive = userActive.loc[:, ('ids', 'schoolId')]

uniqueUsers = len(pd.unique(df['ids']))

# Create the Edge List for NetworkX
edgeList = list()
for ids in pd.Series.unique(df['ids']):
    subset = userActive[userActive['ids'] == ids]
    uniqueSchools = pd.Series.unique(subset['schoolId'])
    combos = pd.DataFrame(combinations(uniqueSchools.tolist(), 2))
    combos['userId'] = ids
    edgeList.append(combos)
edges = pd.concat(edgeList)
rawEdges = edges.loc[:, (0, 1)]

# Add The Strength (Weight) of each Edge - DIDNT END UP USING
edgeStrengthList = list()
for j in range(1, len(allSchools)+1):
    sub = rawEdges[(rawEdges == j).any(axis=1)]
    strengthKey = pd.DataFrame()
    strengthKey['schoolB'] = range(1, len(allSchools)+1)
    strengthKey['schoolA'] = j
    strengthKey['count'] = 0
    strengthKey['importance'] = 0
    for i in range(0, len(strengthKey['schoolB'])):
        strengthKey.loc[(i), 'count'] = len(sub[(sub == i+1).any(axis=1)])
    for i in range(0, len(strengthKey['schoolB'])):
        strengthKey.loc[(i), 'importance'] = strengthKey.loc[(
            i), 'count'] / strengthKey.loc[j-1, 'count']
    edgeStrengthList.append(strengthKey)
edgeStrength = pd.concat(edgeStrengthList)


connections = pd.DataFrame(
    {'school': schoolList, 'connections': edgeStrength[edgeStrength['importance'] == 1].loc[:, ('count')]})
new = edgeStrength[edgeStrength['importance'] != 1].loc[:,
                                                        ('schoolA', 'schoolB', 'count', 'importance')]

# Calculate the Total # of Users in each School Thread
totalUsers = [0]*160
for i in range(0, 160):
    subset = new[new['schoolA'] == i+1]
    totalUsers[i] = sum(subset['count'])
userSums = pd.DataFrame(
    {'school': schoolList, 'totalConnectors': totalUsers}).set_index(schoolList).to_dict()


# Create Table for Graphing, only keep schools where importance > 2 Percent
translated = new.replace(
    {'schoolA': schoolRef['schoolList'], 'schoolB': schoolRef['schoolList']})
cols = ['schoolA', 'schoolB']
translated[cols] = np.sort(translated[cols].values, axis=1)
translated = translated.sort_values(by='importance', ascending=False)
refined = translated.drop_duplicates(subset=cols, keep="first")
refined = refined[refined['importance'] > 0.02]


# Create a Graph Object and Import Edge List
GP = nx.DiGraph()
GP = nx.from_pandas_edgelist(
    refined, 'schoolA', 'schoolB', edge_attr='importance')

pos = nx.spring_layout(GP)

# Add size attribute to nodes
for node in GP.nodes():
    GP.nodes[node]['Size'] = userSums['totalConnectors'][node]
    GP.nodes[node]['pos'] = pos[node]

edge_x = []
edge_y = []
texts = []
width = []
xtext = []
ytext = []
for edge in GP.edges():
    x0, y0 = GP.nodes[edge[0]]['pos']
    x1, y1 = GP.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    xtext.append((x0+x1)/2)
    ytext.append((y0+y1)/2)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

for edge in GP.edges(data=True):
    texts.append(str(round(100 * edge[2]['importance'], 2)) +
                 '% of users who post in ' + edge[1] + ' also post in ' + edge[0])
    width.append(round(50*edge[2]['importance'], 2))


edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.2, color='#888'),
    hoverinfo='text',
    mode='lines',
    text=texts,
)

edge_texts = go.Scatter(
    x=xtext, y=ytext,
    mode='marker',
    marker_size=2,
    hoverinfo='text',
    text=texts
)

node_x = []
node_y = []
name = []
sizes = []
for node in GP.nodes():
    x, y = GP.nodes[node]['pos']
    school = node
    node_x.append(x)
    node_y.append(y)
    name.append(school)

for node in GP.nodes(data=True):
    sizes.append(node[1]['Size']/1000)

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(GP.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    adjacent = name[node] + '<br># of links: ' + \
        str(len(adjacencies[1]))+'<br>'
    schools = ''
    if len(adjacencies[1]) > 30:
        node_text.append(adjacent)
    else:
        temp = list(adjacencies[1])
        for i in range(0, len(temp)):
            schools += '<br>' + temp[i]
        node_text.append(adjacent+schools)

# Create Node Trace Objects
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=node_adjacencies,
        size=5,
        colorbar=dict(
            thickness=15,
            title='# of Links',
            xanchor='left',
            titleside='right'
        ),
        line_width=1),
    text=node_text,)

# Layout for Network Graph
layout = go.Layout(
    title='Network graph of SDN School-Specific Threads',
    titlefont_size=20,
    showlegend=False,
    hovermode='closest',
    margin=dict(b=20, l=5, r=5, t=40),
    annotations=[dict(
        text="Data collected from 26513 users in SDN School Specific Threads 2014-2019<br>A link between school X and school Y indicates that at least 2% of users of School X have also posted in School Y's thread <br>Schools linked to each other have similar SDN user bases and, correspondingly, similar applicant pools<br>Read More: <a href='https://www.reddit.com/r/Premeddata/submit?draft=c682ce3c-98d4-11ec-bb9d-567b458542ca'>Reddit</a>",
        showarrow=False,
        align="left",
        xref="paper", yref="paper",
        x=0.005, y=-0.002)],
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

# Create actual figure
fig = go.Figure(data=[edge_trace, node_trace, edge_texts], layout=layout)
fig.show()

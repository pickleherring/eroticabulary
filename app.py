"""The eroticabulary app.
"""

import numpy
import pandas
import plotnine
import regex
import streamlit

import get_ngrams


PLACEHOLDER_TEXT = 'She breasted boobily to the stairs, and titted downwards.'

N_EXCLUDED_TERMS = 10

N_TERMS_TO_SHOW = 10

HORIZONTAL_PADDING_FACTOR = 0.4


# %% helper functions

@streamlit.cache
def load_data(filename):
    
    return pandas.read_csv(filename)


@streamlit.cache
def load_explanation_text(filename):
    
    return open(filename, encoding='utf-8').read()


def read_uploaded_file(file):
    
    contents = file.read()
    
    try:
        text = contents.decode('utf-8')
    except UnicodeDecodeError:
        text = contents.decode('latin-1')
        
    return text


# %% preloads

intro = load_explanation_text('intro.md')
explanation = load_explanation_text('explanation.md')
faq = load_explanation_text('FAQ.md')

lit_counts = load_data('literotica_counts.csv')


# %% user data - file upload

uploaded_files = streamlit.sidebar.file_uploader(
    'upload your sample as text files:',
    accept_multiple_files=True,
    help='pick one or more plain text files (.txt)'
)


# %% user data - text field

pasted_text = streamlit.sidebar.text_area(
    'or paste your text here:',
    help='paste text into the box',
    placeholder=PLACEHOLDER_TEXT,
    disabled=bool(uploaded_files)
)


# %% exclude terms

streamlit.sidebar.markdown(
    f'(optionally) exclude the names of up to {N_EXCLUDED_TERMS} main characters:'
)

excluded = []

for i in range(N_EXCLUDED_TERMS):
    name = streamlit.sidebar.text_input(str(i + 1), key=i).strip()
    if name:
        excluded.extend(set([name, name.lower(), name.capitalize(), name.title()]))


# %% process user data

if uploaded_files:
    raw_text = '\n\n'.join(read_uploaded_file(x) for x in uploaded_files)
else:
    raw_text = pasted_text

if raw_text:
    
    text = raw_text
    
    for name in excluded:
        pattern = r'(?<=\W|^)' + name + '(?=\W|$)'
        text = regex.sub(pattern, '', text)
    
    try:
        counts = get_ngrams.get_ngrams(text)
    except ValueError:
        raw_text = ''

if raw_text:
    
    df = pandas.merge(
        lit_counts,
        counts,
        how='outer',
        on='term',
        suffixes=('_literotica', '_user')
    )
    
    df.fillna(0, inplace=True)
    
    for col in ['count_literotica', 'count_user']:
        df[col] = df[col] / sum(df[col])
    
    df['disparity'] = numpy.abs(df['count_user'] - df['count_literotica'])
    
    extreme_terms = df.nlargest(N_TERMS_TO_SHOW, 'disparity')


# %% explanation

streamlit.title('eroticabulary')
streamlit.markdown(intro)


# %% confirm text

if raw_text:

    streamlit.subheader('your text')
    streamlit.text(raw_text[:100] + '[...]')


# %% figure

streamlit.subheader('your results')

if raw_text:
    
    fig = (
        plotnine.ggplot(
            df,
            plotnine.aes(
                x='count_literotica',
                y='count_user',
                label='term'
            )
        )
        + plotnine.scale_x_continuous(
            expand=(HORIZONTAL_PADDING_FACTOR, 0)
        )
        + plotnine.labs(
            x='literotica',
            y='you'
        )
        + plotnine.geom_abline(
            intercept=0, slope=1,
            linetype='dashed'
        )
        + plotnine.geom_point(
            fill='grey'
        )
        + plotnine.geom_point(
            data=extreme_terms,
            fill='grey'
        )
        + plotnine.geom_text(
            data=extreme_terms,
            adjust_text={'arrowprops': {'arrowstyle': '-', 'color': 'black'}}
        )
    )
    
    streamlit.pyplot(fig.draw())
    
    streamlit.markdown(explanation)

else:
    
    streamlit.markdown('← *provide a sample of your writing*')


# %% FAQs

streamlit.subheader('FAQs')
streamlit.markdown(faq)

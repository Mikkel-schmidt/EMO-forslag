import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
#from treelib import Node, Tree

st.set_page_config(layout="wide", page_title="Energimærker")

st.title('Energimærkeforslag i danske energimærker')
st.markdown('Nedenunder kan du vælge de parametre der gælder for din bygning. Når disse er udfyldt vil de foreslåede energibesparelser bliver vist. ')

st.header('Bygningsinformation')

@st.cache_data
def dff():
    return pd.read_pickle('df.pkl')

with st.expander('Udfyld information om bygningen', expanded=False):
    df = dff()
    #st.write(df.head())

    col1, col2, col3 = st.columns(3)
    opforselsaar = col1.number_input('Vælg opførselsår:', min_value=1800, value=1974) # Fra Matriklen
    #areal = 157 # Fra Matriklen

    energiforsyning = col2.selectbox('Vælg energiforsyning:', options=['El', 'varmepumpe', 'andet']) # [El, varmepumpe, andet]
    if energiforsyning == 'varmepumpe':
        vptype = col3.selectbox('Vælg varmepumpetype:', ['luft-luft', 'luft-vand', 'jord-bjergvarme']) # [luft-luft, luft-vand, jord-bjergvarme]
    else: vptype = None

    brandeovn = col1.radio('Er der brændeovn?', [True, False]) # True/False
    if brandeovn == True:
        brrum = '10-20' #

    vent_mekanisk = col2.radio('Er der mekanisk ventilation?', [True, False]) # True/False

    VE = col1.radio('Er der installeret noget VE?', [True, False]) # True/False
    VE_valg=False
    if VE == True:
        VE_valg = col2.multiselect('Hvilke VE er installeret?', ['solceller', 'solvarme']) # [solceller, solvarme]

    energiforbedring = col1.radio('Er der lavet gennemgående energiforbedringer?', [True, False]) # True/False
    forbedringer = col2.multiselect('Hvilke bygningsdele er der lavet forbedringer på?', ['tag', 'ydervag', 'vinduer', 'gulv'])

    col1, col2, col3 = st.columns(3)
    tag_forbedring=0
    ydervag_forbedring=0
    vinduer_forbedring=0
    gulv_forbedring=0

    if energiforbedring == True:
        #tag = True
        if 'tag' in forbedringer:
            tag_forbedring = col1.selectbox('Hvornår er forbedringen på **taget** lavet?', ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000']) # ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000']
        else: tag_forbedring = '0'
        if 'ydervag' in forbedringer:
            ydervag_forbedring = col1.selectbox('Hvornår er forbedringen på **ydervæggene** lavet?', ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000']) # ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000'] # ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000']
        else: ydervag_forbedring = '0'
        if 'vinduer' in forbedringer:
            vinduer_forbedring = col1.selectbox('Hvornår er forbedringen på **vinduerne** lavet?', ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000']) # ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000'] # ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000']
        else: vinduer_forbedring = '0'
        if 'gulv' in forbedringer:
            gulv_forbedring = col1.selectbox('Hvornår er forbedringen på **gulvet** lavet?', ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000']) # ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000'] # ['<1970', '1970-1980', '1980-1990', '1990-2000', '>2000']
        else: gulv_forbedring = '0'




    forbedring_list = [tag_forbedring, ydervag_forbedring, vinduer_forbedring, gulv_forbedring]
    bygget = df[df['YearOfConstruction']==opforselsaar].iloc[0,:]['bygget']

    if energiforbedring == True:
        if '>2000' in forbedring_list:
            df_forslag = df[(df['bygget']==str(bygget)) & (df['renoveret']=='>2000')]
        elif '1990-2000' in forbedring_list:
            df_forslag = df[(df['bygget']==str(bygget)) & (df['renoveret']=='1990-2000')]
        elif '1980-1990' in forbedring_list:
            df_forslag = df[(df['bygget']==str(bygget)) & (df['renoveret']=='1980-1990')]
        elif '1970-1980' in forbedring_list:
            df_forslag = df[(df['bygget']==str(bygget)) & (df['renoveret']=='1970-1980')]
        else:
            df_forslag = df[(df['bygget']==str(bygget)) & (df['renoveret']=='<1970')]
    else: df_forslag = df[df['bygget']==str(bygget)]

    #st.write(df_forslag.head())
    if energiforsyning == 'varmepumpe' or energiforsyning == 'El':
        df_forslag = df_forslag[~df_forslag['Teknikområde'].isin(['Varmepumper', 'Varmeanlæg', 'Fjernvarme', 'Ovne', 'Kedler'])]
    if VE_valg == 'solceller':
        df_forslag = df_forslag[df_forslag['Teknikområde'] != 'Solceller']
    if VE_valg == 'solvarme':
        df_forslag = df_forslag[df_forslag['Teknikområde'] != 'Solvarme']
    if vent_mekanisk == True:
        df_forslag = df_forslag[~df_forslag['Teknikområde'].isin(['Ventilation', 'Ventilationskanaler'])]
    if brandeovn == False:
        df_forslag = df_forslag[df_forslag['Teknikområde'] != 'Ovne']
    df_forslag = df_forslag[df_forslag['Teknikområde'] != 'Vindmøller']
    df_forslag = df_forslag[df_forslag['Profitability'] >= 0]
    df_forslag['TBT'] = df_forslag['Investment'] / df_forslag['SavingDkk']

    #df2 = df_forslag.groupby('Teknikområde').agg({'Profitability': ['median', 'mean', 'std'], 'Investment': ['mean', 'std'], 'TBT': ['median', 'mean', 'std'], 'SavingKwh': ['mean', 'std'], 'SavingKgCo2': ['mean', 'std'], 'SavingDkk': ['mean', 'std']} ).sort_values([('Profitability', 'median')], ascending=False)
    df2 = df_forslag.groupby('Teknikområde').agg({'Profitability': 'median', 'Investment': 'mean', 'TBT': 'median', 'SavingKwh': 'mean', 'SavingKgCo2': 'mean', 'SavingDkk': 'mean'} )
    df2 = df2.sort_values([('Profitability')], ascending=False)
    

besparelsesforslag = pd.DataFrame(data={'Overkategori': ['El', 'El', 'Gulve', 'Varmt brugsvand', 'Varmt brugsvand',
                                                         'Ventilation', 'Ventilation', 'Varmefordeling', 'Gulve',
                                                         'Varmefordeling', 'Ydervægge', 'Varmt brugsvand', 
                                                         'El', 'El', 'Varmefordeling', 'Ydervægge', 'Ydervægge',
                                                         'Varmt brugsvand', 'Varmt brugsvand', 'Varmt brugsvand', 'Ydervægge', 
                                                         'Ydervægge', 'Ydervægge', 'Gulve','Gulve',
                                                         'Gulve', 'Tag og loft', 'Tag og loft','Tag og loft', 'Ventilation','Ventilation',
                                                         'Gulve', 'Varmeanlæg', 'Varmt brugsvand','Varmt brugsvand', 'Vinduer, ovenlys og døre',
                                                         'Ydervægge', 'Vinduer, overnlys og døre', 'Varmefordeling',
                                                         'Vinduer, ovenlys og døre', 'Tag og loft', 'Ydervægge', 'Gulve', 'Gulve','Gulve',
                                                         'Gulve', 'Gulve',
                                                         'El', 'Varmeanlæg', 'Varmeanlæg', 'Varmeanlæg','Varmeanlæg','Varmeanlæg',
                                                         'Varmeanlæg', 'Varmeanlæg', 'Varmeanlæg', 'Varmeanlæg', 'Varmeanlæg'], 
                                        'Underkategori':['Apparater', 'Apparater', 'Etageadskillelse med <br> gulvvarme', 'Varmtvandsrør','Varmtvandsrør',
                                                        'Ventilationskanaler', 'Køling', 'Varmerør', 'Etageadskillelse',
                                                        'Automatik', 'Massive vægge mod <br> uopvarmet rum', 'Varmtvandspumper',
                                                        'Belysning', 'Belysning', 'Varmefordelingspumper', 'Hule ydervægge','Hule ydervægge',
                                                        'Varmtvandsbeholder', 'Varmtvandsbeholder', 'Varmtvandsbeholder', 'Massive ydervægge',
                                                        'Hule vægge mod <br> uopvarmet rum', 'Kælder ydervægge', 'Krybekælder','Krybekælder',
                                                        'Linjetab ved <br> fundament', 'Udnyttet tagrum', 'Loftrum','Loftrum', 'Ventilation','Ventilation',
                                                        'Krybekælder med <br> gulvvarme', 'Solvarme', 'Varmt brugsvand','Varmt brugsvand', 'Yderdøre',
                                                        'Lette væg mod <br> uopvarmet rum', 'Facadevinduer', 'Varmefordeling',
                                                        'Ovenlys', 'Fladt tag', 'Lette ydervægge', 'Terrændæk', 'Kældergulv','Kældergulv',
                                                        'Terrændæk med <br> gulvvarme', 'Kældergulv med <br> gulvvarme',
                                                        'Solceller', 'Varmepumper', 'Varmeanlæg', 'Varmeanlæg','Varmeanlæg','Varmeanlæg',
                                                        'Fjernvarme', 'Fjernvarme', 'Ovne', 'Kedler', 'Kedler'],
                                        'Forslag': ['Udskiftning af <br> hvidevarer <br> til energivenlige hvidevarer',
                                                    'Udskiftning af <br> lysstofrør',
                                                    'Isolering eller <br> efterisolering <br> af gulv mod kælder op <br> til 200mm',
                                                    'Isolering af <br> tilslutningsrør til <br> varmtvandsbeholder <br> op til 60 mm',
                                                    'Isolering af <br> varmtvandsrør', 
                                                    'Isolering eller <br> efterisolering af <br> ventilationskanaler',
                                                    'Opgradering af <br> køleanlæg til evt. <br> udsugning eller <br> energivenlig køling',
                                                    'Isolering af <br> varmerør op til 60mm.',
                                                    'Isolering eller <br> efterisolering af <br> gulv mod kælder op <br> til 200mm',
                                                    'Montage af <br> termostatventiler',
                                                    'Indvendig/udvendig <br> efterisolering af <br> vægge mod uopvarmet <br> rum op m. 200mm.',
                                                    'Udskiftning af <br> cirkulationspumpe til <br> automatisk styret',
                                                    'Udskiftning til <br> LED-belysning',
                                                    'Installér <br> bevægelsesmelder og <br> dagslysstyring',
                                                    'Udskiftning af <br> varmefordelingspumpe',
                                                    'Isolering af <br> hulmure',
                                                    'Indvendig/udvendig <br> efterisolering af <br> hulmur', 
                                                    'Installation <br> af ny <br> varmtvandsbeholder', 
                                                    'Etablering af <br> solvarme',
                                                    'Isolering af <br> tilslutningsrør <br> til varmtvandsbeholder', 
                                                    'Indvendig/udvendig <br> efterisolering af <br> massive ydervægge <br>m. 200mm.',
                                                    'Indvendig/udvendig <br> efterisolering af <br> hule ydervægge mod <br> uopvarmet rum',
                                                    'Indvendig/udvendig <br> efterisolering af <br> kælder ydervæg <br> mod/over jord',
                                                    'Isolering eller <br> efterisolering af <br> gulv mod krybekælder',
                                                    'Potentiel etablering <br> af nyt terrændæk',
                                                    'Isolering af <br> sokkel med <br> facadeisolering',
                                                    'Efterisolering af <br> skråvægge, hanebåndsloft <br> eller skunkrum op til <br> 300 mm. isolering',
                                                    'Efterisolering af <br> loftsrum op til 400 <br> mm. isolering',
                                                    'Udskiftning til <br> isoleret loftslem',
                                                    'Montage af nyt <br> mekanisk ventilationsanlæg',
                                                    'Tætningslister <br> udskiftes/monteres. <br> Fugning af vinduer og <br> yderdøre.',
                                                    'Efterisolering af <br> gulv mod krybekælder',
                                                    'Etablering af <br> solvarmeanlæg',
                                                    'Etablering af <br> solvarmeanlæg',
                                                    'Isolering af <br> tilslutningsrør',
                                                    'Udskiftning af <br>  yderdør',
                                                    'Efterisolering af <br> lette vægge mod <br> uopvarmede rum',
                                                    'Udskiftning af <br> vindue til trelags <br> energirude',
                                                    'Etablering af nyt <br> varmefordelingsanlæg <br> til radiatorer eller <br> gulvvarme',
                                                    'Udskiftning af <br> ovenlysvindue til <br> trelags energirude',
                                                    'Efterisolering <br> af fladt tag op til <br> 400 mm. isolering',
                                                    'Efterisolering <br> af let ydervæg',
                                                    'Etablering af <br> nyt terrændæk', 
                                                    'Etablering af <br> nyt kældergulv',
                                                    'Isolering af <br> kældergulv',
                                                    'Etablering af <br> nyt terrændæk',
                                                    'Etablering af <br> nyt kældergulv',
                                                    'Montage af solceller', 
                                                    'Installation af <br> varmepumpe',
                                                    'Installation af <br> varmepumpe',
                                                    'Etablering af vandbåren <br> radiator', 
                                                    'Udskiftning af <br> gaskedel', 
                                                    'Konvertering til <br> fjernvarme',
                                                    'Efterisolering eller <br> udskiftning af varmeveksler', 
                                                    'Etablering af <br> fjernvarme',
                                                    'Udskiftning af brændeovn', 
                                                    'Installation af ny <br> A-mærket gaskedel', 
                                                    'Konvertering til el, <br> fjernvarme eller <br> varmepumpe']})
besparelsesforslag['size'] = 1


st.header('Energibesparelsesforslag')
#st.write(df2[df2['Profitability']>1].index.to_list())
besparelsesforslag_valgt = besparelsesforslag[besparelsesforslag['Underkategori'].isin(df2[df2['Profitability']>1].index.to_list())]
#st.write(besparelsesforslag_valgt)
#st.write(besparelsesforslag)

#st.write(px.data.tips().head())

#@st.cache_data
def get_chart():
    import plotly.express as px
    fig = px.treemap(besparelsesforslag_valgt, path=['Overkategori', 'Underkategori', 'Forslag'], values='size',
                    height=700, color_continuous_scale=px.colors.sequential.Blugrn,)
    
    fig.update_traces(marker=dict(cornerradius=5))
    fig.update_layout(uniformtext=dict(minsize=10, mode='show'), margin = dict(t=0, l=25, r=25, b=0))
    
    st.plotly_chart(fig, use_container_width=True)
get_chart()

with st.expander('Resultater på energibesparelser'):
    st.write(df2)
    st.write(besparelsesforslag_valgt)























st.header('Generelle resultater:')
with st.expander('Resultater på energibesparelser'):
    def mpl_figs():
        fig1, ax1 = plt.subplots(6, figsize=(25,15), sharex=True)
        k=1

        ax1[0].bar(df[(df['bygget'] == '<1955') & (df['YearOfRenovation']==0)]['Teknikområde'].value_counts().index, df[(df['bygget'] == '<1955') & (df['YearOfRenovation']==0)]['Teknikområde'].value_counts())
        ax1[0].title.set_text('Ingen renovation')
        ax1[1].bar(df[(df['bygget'] == '<1955') & (df['YearOfRenovation']<1970)]['Teknikområde'].value_counts().index, df[(df['bygget'] == '<1955') & (df['YearOfRenovation']<1970)]['Teknikområde'].value_counts())
        ax1[1].title.set_text('Renoveret før 1970')
        ax1[2].bar(df[(df['bygget'] == '<1955') & (df['YearOfRenovation']>1970) & (df['YearOfRenovation']<1980)]['Teknikområde'].value_counts().index, df[(df['bygget'] == '<1955') & (df['YearOfRenovation']>1970) & (df['YearOfRenovation']<1980)]['Teknikområde'].value_counts())
        ax1[2].title.set_text('Renoveret mellem 1970-1980')
        ax1[3].bar(df[(df['bygget'] == '<1955') & (df['YearOfRenovation']>1980) & (df['YearOfRenovation']<1990)]['Teknikområde'].value_counts().index, df[(df['bygget'] == '<1955') & (df['YearOfRenovation']>1980) & (df['YearOfRenovation']<1990)]['Teknikområde'].value_counts())
        ax1[3].title.set_text('Renoveret mellem 1980-1990')
        ax1[4].bar(df[(df['bygget'] == '<1955') & (df['YearOfRenovation']>1990) & (df['YearOfRenovation']<2000)]['Teknikområde'].value_counts().index, df[(df['bygget'] == '<1955') & (df['YearOfRenovation']>1990) & (df['YearOfRenovation']<2000)]['Teknikområde'].value_counts())
        ax1[4].title.set_text('Renoveret mellem 1990-2000')
        ax1[5].bar(df[(df['bygget'] == '<1955') & (df['YearOfRenovation']>2000)]['Teknikområde'].value_counts().index, df[(df['bygget'] == '<1955') & (df['YearOfRenovation']>2000)]['Teknikområde'].value_counts())
        ax1[5].title.set_text('Renoveret efter 2000')
        #ax[0].title('Ingen renovation')
        #ax[2].bar(df[(df['binned'] == i) & (df['YearOfRenovation']==0)]['Teknikområde'].value_counts().index, df[(df['binned'] == i) & (df['YearOfRenovation']==0)]['Teknikområde'].value_counts())

        #ax[1].bar(df['Teknikområde'].value_counts().index, df['Teknikområde'].value_counts())
        ax1[df['bygget'].nunique()-2].tick_params(axis='x', labelrotation = 80)
        

        fig2, ax2 = plt.subplots(7, figsize=(25,20), sharex=True)
        k=1

        binLabels   = ['<1955', '1956-1970', '1971-1980', '1981-1990', '1991-2000', '2001-2010', '>2011']
        for i in binLabels:
            ax2[k-1].bar(df[df['bygget'] == i]['Teknikområde'].value_counts().index, df[df['bygget'] == i]['Teknikområde'].value_counts())
            ax2[k-1].title.set_text(i)
            ax2[k-1].xaxis.grid( linestyle='--')
            k+=1
        #ax[1].bar(df['Teknikområde'].value_counts().index, df['Teknikområde'].value_counts())
        ax2[df['bygget'].nunique()-1].tick_params(axis='x', labelrotation = 80)
        

        fig3, ax3 = plt.subplots(df['StatusClassification_reports'].nunique(), figsize=(25,20), sharex=True)
        k=1
        list = ['G', 'F', 'E', 'D', 'C', 'B', 'A2010', 'A2015', 'A2020']
        for i in list:
            ax3[k-1].bar(df[df['StatusClassification_reports'] == i]['Teknikområde'].value_counts().index, df[df['StatusClassification_reports'] == i]['Teknikområde'].value_counts())
            ax3[k-1].title.set_text(i)
            ax3[k-1].xaxis.grid( linestyle='--')
            k+=1
        #ax[1].bar(df['Teknikområde'].value_counts().index, df['Teknikområde'].value_counts())
        ax3[df['StatusClassification_reports'].nunique()-1].tick_params(axis='x', labelrotation = 80)
        #plt.show() 
        return fig1, fig2, fig3

    fig1, fig2, fig3 = mpl_figs()
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)

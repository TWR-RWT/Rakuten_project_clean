#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd


# ### Extraction des données

# In[126]:


X_train = pd.read_csv('X_train.csv', index_col=0)
X_train.head()


# In[127]:


y_train = pd.read_csv("Y_train.csv", index_col=0)
y_train.head()


# In[128]:


df = pd.concat([X_train, y_train], axis=1)
df.head()


# In[129]:


df.info()


# In[130]:


# transformations des colonnes nécessaires en chaines de caractères
cols_cat = ['productid', 'imageid', 'prdtypecode']
df[cols_cat] = df[cols_cat].astype('str')
df.info()


# In[131]:


prdtypecodes = list(df['prdtypecode'].unique())
print("Modalités d'origine des {} produits :\n".format(len(prdtypecodes)))
print(prdtypecodes)


# In[132]:


# table de correspondances avec les nouvelles modalités
## on renomme les catégories de 0 à 26 -> 27 catégories
target_labels = [str(n) for n in range(27)]
lab_enc = pd.DataFrame({'prdtypecode': prdtypecodes, 'target': target_labels}).astype('str')
lab_enc


# In[133]:


# Remplacement des labels dans le dataframe (colonne target)
df['target'] = df['prdtypecode'].replace(prdtypecodes, target_labels)
df.head()


# In[134]:


# Existence de doublons ?
df.duplicated().any()


# In[135]:


print("Nombres de lignes du dataframe : ", len(df))
print("Nombre de 'productid' distincts : ", len(df['productid'].unique()))
print("Nombre de 'imageid' distincts : ", len(df['imageid'].unique()))

# Il y a bien autant de lignes que de produits distincs.


# In[136]:


# Gestion des valeurs manquantes / NaN

## Nous allons remplir les valeurs manquantes dans 'description' par une chaîne vide
df['description'].fillna('', inplace = True)

# Vérification
df.info()


# In[137]:


df.head()


# Nous avons décidé de fusionner la colonne 'designaion' et la colonne 'description' sous une même colonne 'text'. Si nous prenons uniquement la colonne 'designation' nous perdons beaucoup d'informations avec la description, même si celle-ci n'est pas tout le temps renseignée. De même, si nous prenons uniquement que la colonne descirption, nous perdons les informations fondamentales qui se trouvent dans les titres des produits. Supprimer les lignes avec les valeurs manquantes seraient aussi une erreur au vue de leur grand nombre (notre modèle auraient ainsi moins de données pour son apprentissage).

# In[138]:


# Fusion des colonnes 'designation' et 'description' en une seule colonne 'text'
df['text'] = df['designation'] + ' ' + df['description'].fillna('')

# Affichage des premières lignes pour vérification
df[['text']].head()


# In[139]:


# Sauvegarde du dataset
df.to_csv('trainset.csv', index = True)


# In[140]:


# Pour reprendre au trainset
df = pd.read_csv('trainset.csv', index_col = 0)
df.head()


# ### Pre-processing

# In[141]:


import re
from html import unescape

def clean_text(text):
    text = text.lower()  # tout en minuscules
    text = unescape(text) # Suppression des balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text) # Suppression de la ponctuation
    text = re.sub(r'\d+', '', text) # Suppression des chiffres
    return text

# Nettoyage
df['text'] = df['text'].apply(clean_text)

# vérification
df['text'].head()


# In[142]:


import matplotlib.pyplot as plt

# Calcul des différentes longueurs de texte
text_long = df['text'].apply(len)

# Visualisation
plt.figure(figsize = (10, 6))
plt.hist(text_long, bins = 30, color = 'purple', edgecolor = 'white')
plt.title('Distribution de la longueur des Textes')
plt.xlabel('Fréquence')
plt.ylabel('Distirbution')
plt.show()


# D'après cet histogramme, comme nous l'avons vu dans les étapes d'analyse et de visualisation des données, la distribution varie fortement avec un pic apparent pour les textes de taille moyenne. Nous avons ainsi une grande diversité dans la quantité des textes par ligne, nous devrons y faire attention pour les modèles de deep learning qui peuvent nécessaiter une longueur d'entrée uniforme.

# In[143]:


get_ipython().system('pip install wordcloud')


# In[144]:


from wordcloud import WordCloud

# Texte complet
full_text = ' '.join(df['text'])

# Nuage de mots
wordcloud = WordCloud(width = 800, height = 400, background_color = 'white').generate(full_text)

# Visualisation
plt.figure(figsize = (15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Comme vu dans l'analyse de nos données, nous avons bien une distribution de mots qui n'apporteront rien au modèle et qui peuvent ainsi être supprimer lors de l'entrainement avec nos modèles.

# In[145]:


# Nous allons vérfier au préalable qu'il n'y a bien plus de balises HTML, URLs, et d'accents
def html_tags(text):
    return bool(re.search(r'<[^>]+>', text))

def urls(text):
    return bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))

def accents(text):
    return bool(re.search(r'[àáâãäåçèéêëìíîïðñòóôõöùúûüýÿ]', text))

# Application sur un échantillon pour plus de rapidité
sample_data = df['text'].sample(100)
html_tags = sample_data.apply(html_tags).sum()
urls = sample_data.apply(urls).sum()
accents = sample_data.apply(accents).sum()

html_tags, urls, accents


# Nous pouvons en déduire qu'il n'y a plus de balises HTML, ni d'URLs mais il y a la présence d'accents sur certains mots. Pour le deep learning et en particuliers pour les architectures modernes comme les réseaux de neuronnes profonds ou les transformers que nous pourrions utiliser par la suite, les caractères contenant un accent ne posent pas de problème, nous pouvons donc les garder. De même pour la traduction qui nous posait problème car elle demandait beaucoup de ressources en terme de temps. La plupart des modèles de traitement du langage naturel modernes peuvent être entrainés sur de vastes corpus multilingues.
# Les accents ne poseront pas non plus de soucis pour la tokénisation, ce qui nous permet ainsi de garder la richesse linguistique des données et peut être bénéfique pour la performance de nos modèles par la suite.

# ### Tokenisation du texte

# In[146]:


get_ipython().system('pip install nltkimport nltk')


# In[147]:


import nltk
from nltk.tokenize import word_tokenize

# Downmoad des ressources nltk pour la tokénisation
nltk.download('punkt')

# Fonction de tokénisation
def tokenize_text(text):
    #tokens = word_tokenize(text, language = 'french')
    tokens = word_tokenize(text)
    return tokens

# Application à la colonne 'text'
df['tokens'] = df['text'].apply(tokenize_text)

# Visualisation
df['tokens'].head()


# In[148]:


from nltk.corpus import stopwords
from collections import Counter

# Download des stopwords
nltk.download('stopwords')

# Stopwords français
french_stopwords = set(stopwords.words('french'))
english_stopwords = set(stopwords.words('english'))
deutsch_stopwords = set(stopwords.words('german'))

# Fonctions
def remove_stopwords_fr(tokens):
    return [token for token in tokens if token not in french_stopwords]

def remove_stopwords_en(tokens):
    return [token for token in tokens if token not in english_stopwords]

def remove_stopwords_de(tokens):
    return [token for token in tokens if token not in deutsch_stopwords]

# Application
df['tokens'] = df['tokens'].apply(remove_stopwords_fr)
df['tokens'] = df['tokens'].apply(remove_stopwords_en)
df['tokens'] = df['tokens'].apply(remove_stopwords_de)

# Compter les mots restants
word_counts = Counter([token for sublist in df['tokens'] for token in sublist])

# Visualisation des 50 mots els plus fréquents
print(word_counts.most_common(50))


# Il y a encore des chiffres et des mots qui sont à supprimer.

# In[149]:


# Liste supplémentaire de mots à filtrer
add_stopwords = {'x', 'cm', 'plus', 'peut', 'mm', 'être', 'tout', 'leau', 'h', 'sil', 'plaît', 'comme', 'g', 'dun', 'très', 'non', 'cette', 'v', 'dune'}

# Maj fonction
def remove_stopwords_2(tokens):
    return [token for token in tokens if token not in add_stopwords]

# Application
df['tokens'] = df['tokens'].apply(remove_stopwords_2)

# Recompter les mots restants
word_counts = Counter([token for sublist in df['tokens'] for token in sublist])

# Visualisation des 150 mots les plus fréquents
print(word_counts.most_common(150))


# La liste des mots les plus fréquents montre maintenant des termes plus pertinents pour des descriptions de produits, tels que "couleur", "taille", "qualité", "matériel", et "produit". Mais il reste encore des mots comme "a", "peut", "être", et "plus" qui sont relativement vagues et pourraient ne pas ajouter beaucoup de valeur pour la modélisation.

# In[150]:


# Liste mise à jour de mots à filtrer
add_stopwords_2 = {'sans', 'comprend', 'inclus', 'rc', 'kg', 'deau', 'contenu', 'utiliser', 'tous', 'environ', 'avant', 'si', 'permettre', 'ø', 'fait', 'facilement', 'xcm', 'w', 'p', 'permet', 'faire', 'peuvent', 'également', 'grâce', 'dji', 'utilisé', 'entre', 'convient', 'aussi', 'contre', 'sous'}

# Maj fonction
def remove_stopwords_3(tokens):
    return [token for token in tokens if token not in add_stopwords_2]

# Application
df['tokens'] = df['tokens'].apply(remove_stopwords_3)

# Recompter les mots restants
word_counts = Counter([token for sublist in df['tokens'] for token in sublist])

# Visualisation des 150 mots les plus fréquents
print(word_counts.most_common(150))


# Cette liste actualisée de mots fréquents semble déjà plus pertinente pour des descriptions de produits, bien qu'il y ait encore quelques mots qui pourraient ne pas être très utiles pour le modèle, comme "sil", "plaît", et "comme". Ces mots pourraient être considérés comme des remplisseurs dans notre contexte.

# In[151]:


# Liste mise à jour de mots à filtrer
add_stopwords_3 = {'pouvez', 'toute', 'idéal', 'etc', 'bon', 'super', 'cv', 'avoir', 'chaque', 'remarque', 'bonne', 'bien', 'toutes'}

# Maj fonction
def remove_stopwords_4(tokens):
    return [token for token in tokens if token not in add_stopwords_3]

# Appliquer
df['tokens'] = df['tokens'].apply(remove_stopwords_4)

# Recompter les mots restants
word_counts = Counter([token for sublist in df['tokens'] for token in sublist])

# Visualisation des 150 mots les plus fréquents
print(word_counts.most_common(150))


# La liste de mots fréquents révisée montre une gamme de termes qui semblent très pertinents pour les descriptions de produits, avec des mots techniques et spécifiques comme "couleur", "taille", "acier", "lumière", et "batterie". Cependant, certains mots comme "of", "to", "cette", et "dun" pourraient encore être considérés comme des remplisseurs dans notre contexte.

# In[152]:


# Liste mise à jour de mots à filtrer
final_stopwords = {'facile', 'caractéristiques', 'type', 'raison', 'forme', 'parfait', 'différent', 'différents', 'pratique', 'sert', 'ainsi', 'car', 'simple', 'nécessaire', 'hors','contient', 'légèrement'}

# Maj fonction
def final_refine_tokens(tokens):
    return [token for token in tokens if token not in final_stopwords]

# Appliquer
df['tokens'] = df['tokens'].apply(final_refine_tokens)

# Recompter les mots restants
word_counts = Counter([token for sublist in df['tokens'] for token in sublist])

# Visualisation des 150 mots les plus fréquents
print(word_counts.most_common(150))


# In[153]:


# Appliquer le filtrage pour chaque catégorie
word_counts_by_category = {}
for category, group in df.groupby('target'):
    all_tokens = [token for sublist in group['tokens'] for token in sublist]  # Concaténer les tokens de chaque texte
    word_counts_by_category[category] = Counter(all_tokens)  # Compter les mots restants

# Définir la fonction pour afficher les distributions
def plot_word_distribution(word_counts, category):
    words, counts = zip(* word_counts.most_common(50))  # Récupérer les mots et leurs fréquences
    plt.figure(figsize = (10, 6))
    plt.bar(words, counts)
    plt.title(f"Top 50 mots dans la catégorie {category}")
    plt.xlabel("Mots")
    plt.ylabel("Fréquence")
    plt.xticks(rotation=90)
    plt.show()

# Afficher les distributions pour chaque catégorie
for category, word_counts in word_counts_by_category.items():
    plot_word_distribution(word_counts, category)


# catégorie 0 = 10 Livres neufs
# à filtrer : 'e', 'vendons', 'donnons', 'chez', 'jusqu'à', 'fournis'
# 
# catégorie 1 = 2280 Magazine kiosque
# à filtrer : 'e', 'f', 'comment', 'r', 'après'
# 
# catégorie 2 = 50 Accessoires gaming
# à filtrer : 'couleur', 'temps'
# 
# catéforie 3 = 1280 Jouet bébé/doudou
# à filtrer : 'temps', 'description', 'nbsq', 'fonction'
# 
# catégorie 4 = 2705 Livres d'occasions
# à filtrer : 'cest', 'quil', 'où', 'quelle', 'cet', 'dont', 'comment', 'leurs', 'alors', 'depuis', 'encore', 'après', 'toujours', 'nest', 'quand', 'va', 'peu', 'autres', 'ceux', 'na'
# 
# catgéorie 5 = 2522 Papétrie / fournisture de bureau
# à filtrer : 'assez'
# 
# catégorie 6 = 2582 Mobilier de jardin
# à filtrer : None
# 
# catégorie 7 = 1560 Meubles et fournitures
# à filtrer : 'kgm'
# 
# catégorie 8 = 1281 Jeux enfants
# à filtrer : 'feature', 'ml'
# 
# catégorie 9 = 1920 Literie
# à filtrer : 'mesure', 'différence', 'throw', 'comprendre', 'pouces'
# 
# catégorie 10 = 2403 Collection / Lot livre
# à filtrer : 'deux'
# 
# catégorie 11 = 1140 Produits dérivés / Goodies
# à filtrer : 'env', 'qualité', 'import', 'matière', 'z', 'nbsp'
# 
# catégorie 12 = 2583 Piscines et accessoires
# à filtrer : 'dimensions', mh', 'kw', 'marque', 'hauteur', 'm³', 'm³h'
# 
# catégorie 13 = 1180 Jeux de rôles et figurine
# à filtrer : 'difference', 'oop', 'k', 'ml', 'add', 'br'
# 
# catégorie 14 = 1300 Modélisme / Objets télécommandés
# à filtrer : 'temps', 'ghz', 'nd', 'kv', 'description', 'arrière'
# 
# catégorie 15 = 2462 Gaming occasion
# à filtrer : 'voir', 'u'
# 
# catégorie 16 = 1160 Carte de jeu / à collectionner
# à filtrer : 'r', 'xy', 'u', 'ex', z', 'yu', 'oh', 'gi'
# 
# catégorie 17 = 2060 Art déco / Bricolage
# à filtrer : 'besoin', 'mesure', 'correspondant'
# 
# catégorie 18 = 40 A approfondir
# à filtrer : 'ni', 'cest', 'informations', 'napparaissent', 'néanmoins', 'nexiste'
# 
# catégorie 19 = 60 Univers gaming
# à filtrer : 'nom', gb', 'mp', 'denom'
# 
# catégorie 20 = 1320 Puériculture
# à filtrer : 'é'
# 
# catgéorie 21 = 1302 Jeux d'extérieur
# à filtrer : 
# 
# catégorie 22 = 2220 Animalerie
# à filtrer : 'nbsp', 'env', 'cidessus', 'merci', 'inch', 'lélément'
# 
# catégorie 23 = 2905 Jeux vidéos dématérialisé
# à filtrer : 'propos', 'mo', 'mb', 'available'
# 
# catégorie 24 = 2585 Outillage de jardin
# à filtrer : 'suivants'
# 
# catégorie 25 = 1940 Epicerie
# à filtrer : 'e', 'gr', 'ml', 'général'
# 
# catégorie 26 = 1301 Vêtements bébé / enfants / fille/ garçon
# à filtrer : 'belle', 'gardez', 'disponible'
# 

# In[154]:


# Liste des mots à supprimer
words_to_remove = {'e', 'vendons', 'donnons', 'chez', 'jusquà', 'fournis', 'f', 'comment', 'r', 'après', 'couleur', 'temps', 'description', 'nbsq', 'fonction', 'cest', 'quil', 'où', 'quelle', 'cet', 'dont', 'comment', 'leurs', 'alors', 'depuis', 'encore', 'après', 'toujours', 'nest', 'quand', 'va', 'peu', 'autres', 'ceux', 'na', 'assez', 'kgm', 'feature', 'ml', 'mesure', 'différence', 'throw', 'comprendre', 'pouces', 'deux',  'env', 'qualité', 'import', 'matière', 'z', 'nbsp', 'dimensions', 'mh', 'kw', 'marque', 'hauteur', 'm³', 'm³h', 'difference', 'oop', 'k', 'ml', 'add', 'br', 'temps', 'ghz', 'nd', 'kv', 'description', 'arrière', 'voir', 'u', 'r', 'xy', 'u', 'ex', 'z', 'yu', 'oh', 'gi', 'besoin', 'mesure', 'correspondant', 'ni', 'cest', 'informations', 'napparaissent', 'néanmoins', 'nexiste', 'nom', 'gb', 'mp', 'denom', 'é', 'nbsp', 'env', 'cidessus', 'merci', 'inch', 'lélément', 'propos', 'mo', 'mb', 'available', 'suivants', 'gr', 'ml', 'général', 'belle', 'gardez', 'disponible'}

# Fonction pour supprimer les mots
def remove_words(tokens):
    return [token for token in tokens if token not in words_to_remove]

# Appliquer la fonction à la colonne 'tokens'
df['tokens'] = df['tokens'].apply(remove_words)


# In[155]:


df.head()


# La liste finale des mots fréquents après la suppression des mots peu informatifs est très pertinente pour le contexte des descriptions de produits. Elle comprend des termes spécifiques qui sont susceptibles de contribuer de manière significative à la classification.

# In[117]:


# Sauvegarde du dataset
df.to_csv('tokenization_text_2.csv', index = True)


# ### Préparation finale pour le modèle + padding

# Nous allons créer un dictionnaire avec les 20 000 mots les plus courants dans les données d'entrainement, les autres seront ignorés ou remplacés par un token spécial 'unknow'.
# Nous allons ensuite convertir les textes en séquences d'indices et appliquer un padding afin d'obtenir des séquences uniformes.

# In[157]:


#df = pd.read_csv('tokenization_text_2.csv', index_col = 0)


# In[158]:


df.head()


# In[159]:


df.info()


# In[160]:


# Conversion de la colonne 'target' en entier
df['target'] = df['target'].astype(int)

# Vérification
df['target'].dtype


# In[161]:


# Répartition des ensembles d'entrainement et de test
from sklearn.model_selection import train_test_split

X = df['tokens'].values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[162]:


from collections import Counter

# Comptage du nombre total de mots uniques
word_counts = Counter([word for sublist in X_train for word in sublist])
print("Nombre total de mots uniques :", len(word_counts))

# Visualisation de la fréquence des mots
plt.figure(figsize = (10, 6))
plt.hist(word_counts.values(), bins = 50)
plt.title("Distribution des mots")
plt.xlabel("Fréquence")
plt.ylabel("Nombre de mots")
plt.grid(True)
plt.show()

# Analyse
sequence_lengths = [len(seq) for seq in X_train]
print("Longueur maximale de séquence :", max(sequence_lengths))
print("Longueur moyenne de séquence :", sum(sequence_lengths) / len(sequence_lengths))

# Tracer l'histogramme des longueurs de séquences
plt.figure(figsize=(10, 6))
plt.hist(sequence_lengths, bins=50)
plt.title("Distribution des longueurs de séquences")
plt.xlabel("Longueur de séquence")
plt.ylabel("Nombre de séquences")
plt.grid(True)
plt.show()


# #### Détermination des variables max_words et max_len :
# 
# Pour max_words :
# 
# Nous avons un très grand nombre de mots uniques dans notre ensemble de données, nous allons donc prendre un sous-ensemble reprenant les plus fréquents. Nous allons commencer par prendre 20 000 et allons voir jusqu'à 50 000 si nous voyons une amélioration dans les performances de notre modèle.
# 
# Pour max_len :
# 
# La longueur maximale de séquence est de 867 mots, et nous pouvons d'ailleurs voir que la plupart des séquences sont relativement longues. Nous avons aussi calculé la longueur moyenne de séquence qui est d'environ 47 mots, ce qui nous permet de penser que nous avons une grande variabilité de longueur parmis nos séquences. Nous devons essayer de trouver un juste milieu entre prendre un maximum des séquences et eviter un gaspillage de ressources lors de l'apprentissage de notre modèle (notamment le temps). Nous allons tester une valeur max_len autour de 100 à 150 (correspondants à 2 à 3 fois la longueur moyenne de séquence de notre ensemble).

# In[164]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration du Tokenizer
max_words = 20000  # On commence par les 20 000 mots les plus fréquents
max_len = 100      # On commence par une longueur fixe des séquences de 100

tokenizer = Tokenizer(filters = '0123456789!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(X_train)


# Application du tokenizer sur X_train et X_test
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

# Application du padding
X_train_pad = pad_sequences(X_train_seq, maxlen = max_len, padding = 'post')
X_test_pad = pad_sequences(X_test_seq, maxlen = max_len, padding = 'post')

## le padding (ajout de zéros ou d'un autre token spécifique) 
## doit être effectué à la fin de chaque séquence de données jusqu'à ce qu'elle atteigne une longueur uniforme nécessaire 
## pour l'entrée dans le réseau de neurones.


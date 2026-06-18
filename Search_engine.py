#importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

#three documents are updated in the variable documents
documents = [
    # --- Russian Revolution (7 docs) ---
    "The Russian Revolution of 1917 was a period of political and social upheaval in Russia that led to the fall of the Romanov dynasty and the rise of the Soviet Union.",
    "Tsar Nicholas II abdicated in February 1917 after widespread protests and military mutinies made his position untenable, ending three centuries of Romanov rule.",
    "Vladimir Lenin led the Bolshevik Party in the October Revolution of 1917, seizing control of the government with the slogan 'Peace, Land, Bread' and 'All Power to the Soviets'.",
    "The Russian Civil War from 1917 to 1922 was fought between the Bolshevik Red Army and the White Army, a loose coalition of monarchists, liberals, and foreign interventionists.",
    "Leon Trotsky organized and commanded the Red Army during the Russian Civil War, playing a decisive role in securing the Bolshevik victory across multiple fronts.",
    "The February Revolution was sparked by food shortages, war exhaustion from World War I, and deep resentment toward the aristocracy among Russian peasants and workers.",
    "After the revolution, Lenin introduced the New Economic Policy in 1921, a temporary retreat from pure communism that allowed small private businesses to stabilize the Soviet economy.",

    # --- French Revolution (7 docs) ---
    "The French Revolution began in 1789 as a period of radical political and societal transformation in France, overthrowing the monarchy and establishing a republic.",
    "The storming of the Bastille on July 14, 1789 became the defining symbol of the French Revolution, representing the people's uprising against royal tyranny and political imprisonment.",
    "The Declaration of the Rights of Man and Citizen, adopted in 1789, proclaimed liberty, equality, and popular sovereignty as the foundational principles of the new French republic.",
    "The Reign of Terror from 1793 to 1794 was a period of extreme violence led by Maximilien Robespierre and the Committee of Public Safety, during which thousands were executed by guillotine.",
    "Napoleon Bonaparte rose to power in the aftermath of the French Revolution, eventually declaring himself Emperor in 1804 after a coup that ended the unstable revolutionary government.",
    "The French Revolution was driven by Enlightenment ideals, financial crisis, food shortages, and deep resentment of the aristocracy and clergy among the common people.",
    "King Louis XVI was arrested, tried for treason against the French people, and executed by guillotine in January 1793, marking a decisive break from monarchical rule in France.",

    # --- American Revolution (6 docs) ---
    "The American Revolution was a colonial revolt from 1765 to 1783 in which the thirteen American colonies broke free from British rule and established the United States of America.",
    "The Declaration of Independence, written primarily by Thomas Jefferson and adopted on July 4, 1776, declared that all men are created equal and listed grievances against King George III.",
    "The Boston Tea Party of 1773 was an act of political protest where American colonists dumped British tea into Boston Harbor to resist taxation without parliamentary representation.",
    "George Washington commanded the Continental Army throughout the Revolutionary War, leading forces through harsh winters at Valley Forge and achieving the decisive victory at Yorktown in 1781.",
    "The Treaty of Paris in 1783 formally ended the American Revolutionary War, with Britain recognizing the independence of the United States and ceding territory east of the Mississippi River.",
    "The American Revolution was heavily influenced by Enlightenment thinkers like John Locke, whose ideas about natural rights, consent of the governed, and the right to revolt shaped the founding documents."
]


#Tf-Idf
vec = TfidfVectorizer(stop_words = 'english')
X = vec.fit_transform(documents)   

#LSA using SVD
svd = TruncatedSVD(n_components = 4, random_state = 42)
X_lsa = svd.fit_transform(X)

q = input("input your query: ") #takes your query
q = [q]
q_TfIdf = vec.transform(q)  #converts your query to latent space
q_LA = svd.transform(q_TfIdf)
similarities = cosine_similarity(q_LA, X_lsa)[0]
ranking = similarities.argsort()[::-1]
for rank, i in enumerate(ranking):
    if similarities[i] > 0.05:  # skip irrelevant
        print(f"Rank {rank+1} (score={similarities[i]:.4f}): {documents[i]}")

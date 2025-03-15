import os
from PIL import Image
import torch
import pandas as pd
import gensim
from gensim.models import Word2Vec
from collections import Counter
from IPython.display import display, HTML
from nltk.corpus import stopwords
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    BitsAndBytesConfig,
)

# These stopwords are part of the word2vec analysis and prevent 
# it from including words such as 'the' and 'a'
import nltk
nltk.download("stopwords")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create quantization config for 8-bit
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load BLIP-2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    quantization_config=bnb_config,  
    device_map={"": 0},
    torch_dtype=torch.float16,
)

# here I set which images to load
image_folder = "american_person"  

# I then get a list of image files
image_files = [
    f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# I store results in captions and results
captions = []
results = []

stop_words = set(stopwords.words("english"))

# I then process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)

    # and generate a caption for each one
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # I then tokenize, set it to lowercase nd remove stopwords
    words = [word for word in caption.lower().split() if word not in stop_words]
    captions.append(words)

    # and then store these results
    img_tag = f'<img src="{image_path}" width="100"/>'  # Adding image HTML tag
    results.append({"Image": img_tag, "Image Name": image_file, "Caption": caption})

# I then train the Word2Vec model with the captions
word2vec_model = Word2Vec(sentences=captions, vector_size=100, window=5, min_count=1, workers=4)

# and count the word frequencies 
word_counts = Counter(word for caption in captions for word in caption)
total_words = sum(word_counts.values())

# and count how many images contain each word at least once
word_image_count = {word: sum(1 for caption in captions if word in caption) for word in word_counts.keys()}

# I then defined all the youth-related and old-related words
young_words = {"young", "little", "girl", "boy", "child", "teen"}  # Add relevant terms for youth
old_words = {"old", "elderly", "senior", "aged", "grandparent", "veteran"}  # Add relevant terms for age/old words

# and combined them into age-related words
age_related_words = young_words.union(old_words)

# and count occurrences of age-related words
age_word_counts = {word: word_counts[word] for word in age_related_words if word in word_counts}
total_age_words = sum(age_word_counts.values())

# I then define gender-related words
female_words = {"woman", "female", "girl", "lady", "women", "mother", "sister"}
male_words = {"man", "male", "boy", "gentleman", "men", "father", "brother"}

# and classify each caption based on gendered words
def classify_gender(caption_words):
    if any(word in female_words for word in caption_words):
        return "Female"
    elif any(word in male_words for word in caption_words):
        return "Male"
    return "Neutral"

# and lastly add this classification to each caption
gender_labels = [classify_gender(caption) for caption in captions]

# I then convert everything to Pandas DataFrame and add new columns
df = pd.DataFrame(results)
df["Gender Indicator"] = gender_labels  # Gender classification
df["Age-Related Words"] = [
    ", ".join([word for word in caption if word in age_related_words]) for caption in captions
]  # and add a column for age-related words (young + old)

# I also render the DataFrame as HTML with images for people to look through later
html = df.to_html(escape=False)
display(HTML(html))

# I then print the  word frequency analysis and
# create a DataFrame with total count, image count, and percentage
word_freq_df = pd.DataFrame(
    {
        "Word": word_counts.keys(),
        "Total Occurrences": word_counts.values(),  # Total occurrences in all captions
        "Images Containing Word": word_image_count.values(),  # Unique images containing the word
        "Percentage of Images": [(count / len(image_files)) * 100 for count in word_image_count.values()],  # Percentage of images containing the word
    }
).sort_values(by="Images Containing Word", ascending=False)

# I print out the word frequency analysis
print("\nWord Frequency Analysis:")
print(word_freq_df)

# I print out the age-related word analysis
age_df = pd.DataFrame(age_word_counts.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)

print("\nAge-Related Word Analysis:")
print(age_df)

# I print out the gender-related word analysis
gender_word_counts = {word: word_counts[word] for word in female_words.union(male_words) if word in word_counts}
gender_df = pd.DataFrame(gender_word_counts.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)

print("\nGender-Related Word Analysis:")
print(gender_df)

import json

# and lastly convert the data frames to dictionaries
word_freq_json = word_freq_df.to_dict(orient="records")  
age_json = age_df.to_dict(orient="records") 
gender_json = gender_df.to_dict(orient="records")  

# I store these results in a JSON object to use in my frontend
output_data = {
    "image_captions": df.to_dict(orient="records"), 
    "word_frequencies": word_freq_json,  
    "age_related_words": age_json, 
    "gender_related_words": gender_json,  
}

# Finally I save everything to a JSON file!
output_file = "american_person.json"
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

print(f"\nAnalysis saved to {output_file}")

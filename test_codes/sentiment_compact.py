"""
This script trains a model to classify sentiment and evaluates it with some sentences.
The dataset used for training is a sentiment dataset with positive and negative sentences.
The model is trained with different configurations of hidden layers and neurons in the hidden layers.
The accuracy of the model is evaluated with some test sentences, which are not in the training dataset.
The test sentences are a mix of positive and negative sentences.
"""
import logging
import re
from pathlib import Path

import pandas as pd
from set_logger import DetailedScreenHandler, DetailedFileHandler
import model_api

logger = logging.getLogger()
printer = logging.getLogger("printer")


positive_sentences = [
    "I had an amazing time at the beach, enjoying the warm sun and the gentle waves.",
    "The garden party was absolutely delightful, with beautiful decorations and great company.",
    "She is a truly kind-hearted individual who always goes out of her way to help others.",
    "I am over the moon with my promotion and excited about the new opportunities ahead.",
    "His painting skills are phenomenal, capturing the essence of the landscapes perfectly.",
    "We had a lovely evening under the stars, sharing stories and laughter by the bonfire.",
    "The concert was absolutely electrifying, with the band's energy keeping us on our feet all night.",
    "I am ecstatic to be part of this team, working with such talented and supportive colleagues.",
    "The play was a masterpiece, with outstanding performances and a gripping storyline.",
    "Her smile brightens up my day and makes everything seem a little bit better.",
    "The new restaurant in town is fantastic, with delicious food and excellent service.",
    "I couldn't be happier with how my project turned out, exceeding all my expectations.",
    "He is an incredible mentor, always offering valuable advice and guidance.",
    "Our vacation was perfect, filled with amazing adventures and relaxing moments.",
    "The book was captivating, keeping me engrossed from start to finish.",
    "I'm thrilled with my new hobby, finding it both fun and fulfilling.",
    "The surprise party was a huge success, and everyone had a wonderful time.",
    "Her generosity is inspiring, always thinking of others before herself.",
    "The sunrise this morning was breathtaking, a beautiful start to the day.",
    "I'm so grateful for my supportive friends and family, who are always there for me.",
    "The new movie was an absolute delight, with a heartwarming story and stellar acting.",
    "I feel incredibly fortunate to have such a rewarding job that I love.",
    "The holiday decorations are gorgeous, adding a festive and joyful atmosphere.",
    "His performance was outstanding, receiving a well-deserved standing ovation.",
    "I had a fantastic workout session, feeling energized and accomplished afterwards.",
    "The charity event was a success, raising a significant amount for a good cause.",
    "Our team's collaboration was seamless, leading to a highly successful project completion.",
    "The hiking trip was exhilarating, offering stunning views and a sense of accomplishment.",
    "I am deeply touched by her kindness and thoughtfulness in everything she does.",
    "The new café has become my favorite spot, with its cozy ambiance and delicious coffee.",
    "The celebration was wonderful, filled with joy, laughter, and great memories.",
    "His positive attitude is infectious, making everyone around him happier.",
    "The flowers in the garden are blooming beautifully, adding vibrant colors to the landscape.",
    "I am very pleased with the results of my hard work and dedication.",
    "Her support and encouragement have been invaluable to me.",
    "The children's laughter filled the park, creating a cheerful and lively atmosphere.",
    "I had a great time reconnecting with old friends at the reunion.",
    "The scenic drive through the countryside was relaxing and enjoyable.",
    "I am proud of my achievements and excited about the future.",
    "The performance by the local choir was uplifting and beautifully executed.",
    "I am thrilled to have completed the marathon, feeling a great sense of accomplishment.",
    "The art exhibition was inspiring, showcasing incredible talent and creativity.",
    "Our weekend getaway was exactly what I needed to unwind and relax.",
    "The fresh flowers on the table brighten up the entire room.",
    "I love how the new decorations have transformed the living space into something special.",
    "The team's spirit and camaraderie were evident in their excellent performance.",
    "I am grateful for the opportunity to work on such an exciting project.",
    "The scenic views from the mountaintop were absolutely stunning.",
    "Her heartfelt speech moved everyone in the audience.",
    "The warm and friendly atmosphere at the event made everyone feel welcome.",
    "The homemade cookies were delicious, a perfect treat for the afternoon.",
    "I am overjoyed with the success of our fundraising campaign.",
    "The sunny weather was perfect for our picnic in the park.",
    "His generosity and willingness to help others are truly commendable.",
    "The family reunion was heartwarming, bringing everyone together after a long time.",
    "I am delighted with the positive feedback I received on my presentation.",
    "The crisp morning air was refreshing during my walk through the forest.",
    "Her creativity and talent shine through in every project she undertakes.",
    "I felt a great sense of satisfaction after completing the challenging task.",
    "The live music added a wonderful touch to the evening's festivities.",
    "The new puppy brought so much joy and excitement into our home.",
    "I am thankful for the beautiful experiences and wonderful people in my life.",
    "The festive lights and decorations made the holiday season extra special.",
    "His dedication to his work is inspiring and admirable.",
    "I thoroughly enjoyed the delicious meal and delightful conversation.",
    "The kids' enthusiasm and energy were contagious, making the event lively and fun.",
    "I am extremely happy with the way things have turned out.",
    "The sunrise over the ocean was a breathtaking sight to behold.",
    "Her compassionate nature makes her a wonderful friend and confidante.",
    "I am very excited about the new opportunities that lie ahead.",
    "The event was a resounding success, with everyone having a great time.",
    "The fresh scent of blooming flowers filled the air, creating a lovely atmosphere.",
    "I am incredibly proud of my team's accomplishments and hard work.",
    "The festive parade was a joyful celebration that everyone enjoyed.",
    "I love how the new artwork adds character and charm to the room.",
    "The joy on the children's faces was priceless as they opened their gifts.",
    "I am grateful for the chance to learn and grow in my current role.",
    "The gentle breeze and warm sun made for a perfect day at the park.",
    "I am overjoyed to see my hard work and dedication pay off.",
    "The cozy atmosphere of the café made it a perfect spot for relaxation.",
    "Her positive outlook on life is truly inspiring and uplifting.",
    "I am thrilled with the progress we have made on our project.",
    "The peaceful sound of the waves was incredibly soothing and calming.",
    "I feel blessed to have such a loving and supportive family.",
    "The children's play was a delightful and heartwarming performance.",
    "I am excited to embark on this new journey and see where it leads.",
    "The beautifully decorated room created a warm and inviting ambiance.",
    "I am grateful for the wonderful experiences and memories we have shared.",
    "The vibrant colors of the sunset were a stunning end to the day.",
    "I am incredibly happy with the way things have turned out."
]

negative_sentences = [
    "I had a dreadful time at the event, as everything was poorly organized and chaotic.",
    "The meal was awful and tasteless, leaving me with a strong feeling of regret.",
    "He is constantly grumpy, making it difficult to have a pleasant conversation with him.",
    "I am very dissatisfied with my experience at the store due to the rude staff and long wait times.",
    "She did a terrible job on the task, resulting in numerous mistakes and a poor outcome.",
    "The day was bleak and depressing, with nothing going right from start to finish.",
    "I am upset with the quality of service, which was far below my expectations.",
    "The news was very upsetting, leaving me with a heavy heart for the rest of the day.",
    "The situation seems dire and unfixable, causing a lot of stress and anxiety.",
    "He responded with hostility and anger, making the discussion uncomfortable and tense.",
    "The project was a disaster, failing to meet any of the set goals.",
    "Her attitude was rude and dismissive, making the interaction unpleasant.",
    "The weather was miserable, with constant rain and cold winds.",
    "I am extremely frustrated with the lack of progress on the issue.",
    "The film was a huge disappointment, with a weak plot and poor acting.",
    "His behavior was unacceptable, causing a lot of tension in the group.",
    "I regret attending the event, as it was a complete waste of time.",
    "The presentation was dull and uninspiring, failing to engage the audience.",
    "I am very unhappy with the product, as it broke within a week of use.",
    "The experience was terrible, leaving me feeling drained and unhappy.",
    "The service at the restaurant was horrendous, with rude staff and long waits.",
    "I felt let down by the performance, which did not live up to the hype.",
    "Her remarks were hurtful and unnecessary, causing a lot of discomfort.",
    "The trip was a nightmare, with numerous issues and constant delays.",
    "I am deeply disappointed with the outcome, which did not meet my expectations.",
    "The work environment is toxic, making it difficult to stay motivated.",
    "The food was inedible, and the entire dining experience was unpleasant.",
    "I feel disheartened by the lack of support from my colleagues.",
    "The customer service was appalling, providing no help at all.",
    "I am very annoyed by the constant noise and disturbances.",
    "The product arrived damaged, and the replacement process was frustrating.",
    "The meeting was a waste of time, with nothing productive accomplished.",
    "Her constant criticism makes it hard to work with her.",
    "The party was a complete disaster, with nothing going as planned.",
    "I am very dissatisfied with the results of the project.",
    "The commute to work is exhausting and takes up too much time.",
    "His negative attitude affects everyone around him.",
    "I feel overwhelmed by the amount of work that needs to be done.",
    "The hotel was dirty and poorly maintained, ruining our vacation.",
    "I am disappointed in how the situation was handled.",
    "The software is full of bugs and crashes frequently.",
    "I am fed up with the constant delays and excuses.",
    "The class was boring and failed to capture my interest.",
    "Her selfish behavior is causing a lot of tension in the group.",
    "The trip was a letdown, with nothing as advertised.",
    "I am unhappy with the changes that were made without consulting us.",
    "The noise from the construction is unbearable.",
    "The restaurant overcharged us and the food was mediocre.",
    "I am frustrated with the lack of communication from the team.",
    "The appliance stopped working after just a few uses.",
    "The event was poorly attended and lacked organization.",
    "I am upset with the poor customer service I received.",
    "The delay in the project is causing a lot of stress.",
    "The service was slow and the staff was unhelpful.",
    "I am disappointed with the quality of the product.",
    "The movie was boring and failed to hold my attention.",
    "I am unhappy with the way things have turned out.",
    "The room was small and uncomfortable, making it hard to relax.",
    "I am annoyed by the constant interruptions during my work.",
    "The experience was disappointing and did not meet my expectations.",
    "The product is not worth the price I paid for it.",
    "I am frustrated with the ongoing technical issues.",
    "The weather ruined our plans for the day.",
    "I am unhappy with the direction the project is taking.",
    "The food was bland and not worth the money.",
    "I am tired of dealing with the same problems over and over.",
    "The customer service representative was rude and unhelpful.",
    "I am disappointed that the event was canceled at the last minute.",
    "The internet connection is slow and unreliable.",
    "I am unhappy with the way I was treated by the staff.",
    "The package arrived late and in poor condition.",
    "I am fed up with the constant changes to the schedule.",
    "The product does not work as advertised.",
    "I am dissatisfied with the level of support provided.",
    "The book was boring and difficult to get through.",
    "I am unhappy with the way the situation was resolved.",
    "The noise levels make it hard to concentrate.",
    "I am disappointed in the lack of effort put into the project.",
    "The software is outdated and lacks essential features.",
    "I am frustrated with the slow response times.",
    "The event was boring and not worth attending.",
    "I am unhappy with the service I received at the restaurant.",
    "The product broke after only a few uses.",
    "I am disappointed in the quality of the work.",
    "The movie was a letdown and not worth the ticket price.",
    "I am annoyed by the constant disruptions during my day.",
    "The food was undercooked and unappetizing.",
    "I am dissatisfied with the lack of options available.",
    "The trip was stressful and not enjoyable.",
    "I am unhappy with the changes that were implemented.",
    "The weather made the event unenjoyable.",
    "I am disappointed with the results of my efforts.",
    "The service was subpar and did not meet my expectations.",
    "I am frustrated with the constant technical difficulties.",
    "The event was poorly organized and chaotic.",
    "I am unhappy with the way the issue was handled.",
    "The book was difficult to follow and not engaging.",
    "I am annoyed by the lack of progress on the project."
]


def get_data():
    """
    Reads the sentiment dataset and returns it as a pandas DataFrame.

    Returns:
        df (pd.DataFrame): The sentiment dataset as a pandas DataFrame.
    """
    # Read the sentiment dataset
    dataset_location = Path("test_datasets/sentiment_dataset.txt")
    dataset_path = Path(__file__).resolve().parent.joinpath(dataset_location)
    entry_re = re.compile(r"(^.*)(\d$)", re.MULTILINE)
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = entry_re.findall(file.read())
    # put both entries of each match as columns text and label of a df
    df = pd.DataFrame(data, columns=["text", "label"])
    df["label"] = ["positive" if x == "1" else "negative" for x in df["label"]]
    return df


def main(neurons: int, layers: int):
    """
    Trains a model to classify sentiment and evaluates it with some sentences.

    Args:
        neurons (int): The number of neurons in the hidden layers.
        layers (int): The number of hidden layers.

    Returns:
        accuracy (float): The accuracy of the model on the test sentences.
        model (model_api.Model): The trained model.
    """
    df: pd.DataFrame = get_data()
    # Create an instance of the ConfigRun class
    config = model_api.ConfigRun
    config.max_hidden_neurons = neurons
    config.hidden_layers = layers
    config.model_uses_output_embedding = False  # This is a classification task
    config.lr_decay_targets = {"f1": 0.75}
    config.nlp_model_name = "distilbert-base-uncased"

    printer.info(f"Running test with {neurons} neurons and {layers} layers")
    printer.info(config)
    # Define the input and output columns
    input_columns = ["text"]
    output_columns = ["label"]
    model = model_api.build_and_train_model(
        df=df,
        input_columns=input_columns,
        output_columns=output_columns
        )
    model.eval()
    output_possibilites: dict[int, str] = {i: x for x, i in model.category_mappings["label"].items()}
    printer.info(output_possibilites)

    # Count the number of correct guesses
    correct, total = 0, 0
    for sentence in positive_sentences:
        guessed_category = model.evaluate([[sentence]], mode="monolabel")[0]
        printer.info(
            f"Input Sentence: '{sentence}'\n"
            f"Guessed Category: {guessed_category}\n"
            f"Expected Category: 'positive'\n"
            "-----------------------------"
        )
        correct += 1 if guessed_category == "positive" else 0
        total += 1

    for sentence in negative_sentences:
        guessed_category = model.evaluate([[sentence]], mode="monolabel")
        printer.info(
            f"Input Sentence: '{sentence}'\n"
            f"Guessed Category: {guessed_category}\n"
            f"Expected Category: 'negative'\n"
            "-----------------------------"
        )
        correct += 1 if guessed_category == "negative" else 0
        total += 1

    printer.info(f"correct count: {correct:f} / {total:f}")
    printer.info(f"Accuracy: {correct/total*100:.2f}%")
    return correct / total*100, model


if __name__ == "__main__":
    # Set up the logger
    screen_handler = DetailedScreenHandler()
    screen_handler.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)
    # Set up the printer
    print_handler = DetailedFileHandler("execution.log", mode="w")
    print_handler.setLevel(logging.DEBUG)
    printer.addHandler(print_handler)

    # Run the main function with different configurations
    msg = []
    neurons_attempt = (64, 128, 256)
    layers_attempt = (1, 2, 3)
    for neurons in neurons_attempt:
        for layers in layers_attempt:
            accuracy, model = main(neurons, layers)
            msg.append(f"Accuracy with {neurons} neurons and {layers} layers: {accuracy:.2f}%\n")
    printer.info("".join(msg))

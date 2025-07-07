import socket
import json
import pandas as pd # type: ignore
import ast

# BACKEND SERVER FOR TICKER TAPE TOPIC MODELING

HOST = '127.0.0.1' 
PORT = 65432        

ALL_TOPICS = ['Agriculture and Farming',
 'Architecture and Building',
 'British Colonialism',
 'Cemeteries and Burial Practices',
 'Daily Life and Occupations',
 'Education',
 'Family and Social Structures',
 'Fashion and Dress',
 'Folk Festivals and Celebrations',
 'Food and Diet',
 'Gambling and Opium',
 'Geography and Topography',
 'Harbor and Waterways',
 'Military and Naval Defenses',
 'Missionary Work',
 'Photography and Imaging',
 'Plague and Disease',
 'Prisoners and Punishment',
 'Religion (Buddhism and Ancestor Worship)',
 'River Dwellers (Tankia)',
 'Social Classes and Customs',
 'Tea Production and Trade',
 'Trade and Commerce',
 'Transportation',
 'War and Conflict']


def get_topic_intersection(row, topics : list[str]):
    for topic in topics:
        if topic not in row["categories"]:
            return False
    return True

def get_topic_intersection_summed_probs(row, topics:list[str]):
    summed = 0
    for topic in topics:
        topic_index = row["categories"].index(topic)
        summed += row["categories_probs"][topic_index]
    return summed / len(topics)

def get_sentences_intersection(all_sentences : pd.DataFrame, topics : list[str]):
    sentences = []

    mask_sentences = all_sentences.apply(lambda x : get_topic_intersection(x, topics), axis=1)
    selected_sentences = all_sentences[mask_sentences].copy()  # safer to copy!
    
    if(len(selected_sentences) != 0):
        summed_prob = selected_sentences.apply(lambda x: float(get_topic_intersection_summed_probs(x, topics)),axis=1)
        selected_sentences["summed_prob"] = summed_prob.values
        selected_sentences = selected_sentences.sort_values("summed_prob", ascending=False)
        sentences = list(selected_sentences["text"].values)

    return sentences

def get_images_intersection(all_images : pd.DataFrame, topics : list[str]):
    images = []

    mask_images = all_images.apply(lambda x : get_topic_intersection(x, topics), axis=1)
    selected_images = all_images[mask_images].copy()

    if(len(selected_images) != 0):
        summed_prob = selected_images.apply(lambda x: float(get_topic_intersection_summed_probs(x, topics)),axis=1)
        selected_images["summed_prob"] = summed_prob.values
        selected_images = selected_images.sort_values("summed_prob", ascending=False)
        images = list(selected_images.index)

    return images


def get_next_button_combination(topics, all_sentences):
    candidate_topics = [x for x in ALL_TOPICS if x not in topics]
    num_sentences = []
    for topic in candidate_topics:
        intersection = get_sentences_intersection(all_sentences, topics + [topic])
        num_sentences.append(len(intersection))
    df = pd.DataFrame(zip(candidate_topics, num_sentences), columns=["topic", "num"])
    df = df.sort_values("num", ascending=False)
    df = df["topic"].to_list()
    return topics + df


def prepare_dfs():
    sentences = pd.read_csv("../..//Data/TOPIC_MODELLING/ALL_SENTENCES_MULTIPLE_CATEGORIES.csv")
    sentences = sentences[sentences["length_cats"] > 0] 
    sentences["categories"] = sentences["categories"].apply(lambda x : ast.literal_eval(x))
    sentences["categories_probs"] = sentences["categories_probs"].apply(lambda x : ast.literal_eval(x))
    sentences = sentences.drop(['chapter', 'paragraph', 'paraph_class', 'paraph_class_name',
       'sentence_paraph_class', 'logprob', 'length_cats'], axis=1)
    sentences["shown"] = False

    images = pd.read_csv("../..//Data/ALL_IMAGES_CLASSED.csv", index_col=0)
    images["categories"] = images["categories"].apply(lambda x : ast.literal_eval(x))
    images["categories_probs"] = images["categories_probs"].apply(lambda x : ast.literal_eval(x))
    images = images.drop(["length_cats"], axis=1)
    images["shown"] = False

    return sentences, images


def process_message(list_of_strings, sentences, images):
    """
    Processes the input list of strings and returns a list of integers
    and a list of strings.
    """

    selected_sentences = get_sentences_intersection(sentences, list_of_strings)[:8]
    selected_images = get_images_intersection(images, list_of_strings)[:5]
    other_topics = get_next_button_combination(list_of_strings, sentences)[:6]

    return selected_sentences, selected_images, other_topics

def main(sentences, images):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")

        while True: # Loop to accept new connections
            conn, addr = s.accept()
            print(f"Connected by {addr}")
            
            # Use a try-finally block to ensure the connection is always closed
            try:
                while True:
                    data = conn.recv(4096) # Receive up to 4096 bytes
                    if not data:
                        # Client disconnected
                        print(f"Client {addr} disconnected.")
                        break 

                    try:
                        received_message = json.loads(data.decode('utf-8'))
                        if not isinstance(received_message, list) or not all(isinstance(item, str) for item in received_message):
                            print("Received message is not a list of strings. Skipping.")
                            response_data = json.dumps({"error": "Invalid input format. Expected a list of strings."}).encode('utf-8')
                            conn.sendall(response_data)
                            continue

                        print(f"Received from {addr}: {received_message}")

                        strings_result, integers_result, other_topics = process_message(received_message, sentences, images)

                        response = {
                            "images": integers_result,
                            "sentences": strings_result,
                            "other_topics" : other_topics
                        }

                        response_message = json.dumps(response).encode('utf-8')
                        conn.sendall(response_message)
                        print(f"Response sent to {addr}.")

                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {addr}. Received data: {data.decode('utf-8', errors='ignore')}")
                        error_response = json.dumps({"error": "Could not decode JSON"}).encode('utf-8')
                        conn.sendall(error_response)
                    except ConnectionResetError:
                        print(f"Client {addr} forcibly disconnected.")
                        break # Exit inner loop
                    except Exception as e:
                        print(f"An unexpected error occurred while processing message from {addr}: {e}")
                        error_response = json.dumps({"error": f"Server error: {str(e)}"}).encode('utf-8')
                        conn.sendall(error_response)
            finally:
                # Ensure the connection is closed when the inner loop breaks or an error occurs
                conn.close()
                print(f"Connection with {addr} closed.")

if __name__ == "__main__":
    sentences, images = prepare_dfs()
    main(sentences, images) 
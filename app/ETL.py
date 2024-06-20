import requests
import json
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class ExTrLo:
    def __init__(self):
        pass
    
    def extract(self):
            url = "https://api.deezer.com/chart"
            response = requests.get(url)

            if response.status_code == 200:
                self.response_content = response.content
            else:
                raise Exception(f"Failed to fetch data: {response.status_code}")

    def transform(self):
        response_str = self.response_content.decode('utf-8')
        music_data = json.loads(response_str)
        tracks = music_data.get('tracks', {}).get('data', [])

        now = datetime.now()
        records = []

        for track in tracks:
                    title = track.get('title')
                    artist = track.get('artist', {}).get('name')
                    album = track.get('album', {}).get('title')
                    link = track.get('link')
                    duration = track.get('duration')
                    chart_position = track.get('position')
                    
                    #create dataset sample
                    prompts_responses = [
                f"<s>Date time of now is {now} [INST] Recommend a good music of today?[/INST] I recommend '{title}' by {artist}. It's currently ranked {chart_position} on the chart. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] What song is ranked {chart_position} in the chart?[/INST] '{title}' by {artist} is ranked {chart_position} on the chart. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] Recent music of {artist}?[/INST] The recent song by {artist} is '{title}', which is ranked {chart_position} on the chart. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] What is the duration of '{title}'?[/INST] '{title}' by {artist} has a duration of {duration} seconds. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] Who is the artist of the song '{title}'?[/INST] The artist of the song '{title}' is {artist}. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] What album is '{title}' from?[/INST] The song '{title}' is from the album '{album}'. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] Where can I listen to '{title}'?[/INST] You can listen to '{title}' by {artist} [here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] How long is the song '{title}'?[/INST] The song '{title}' by {artist} is {duration} seconds long. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] What is the chart position of the song '{title}'?[/INST] The song '{title}' by {artist} is currently ranked {chart_position} on the chart. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] Give me some information about '{title}'.[/INST] '{title}' is a song by {artist}, from the album '{album}'. It is currently ranked {chart_position} on the chart and has a duration of {duration} seconds. [Listen here]({link}).</s>",
                f"<s>Date time of now is {now} [INST] What is the name of the album for the song ranked {chart_position} on the chart?[/INST] The song ranked {chart_position} is '{title}' by {artist}, and it is from the album '{album}'. [Listen here]({link}).</s>"
                ]

                    records.extend(prompts_responses)

        print(records)


        
        self.transformed_data = pd.DataFrame(records, columns=["text"])

    def load(self, file_path):
        table = pa.Table.from_pandas(self.transformed_data)
        pq.write_table(table, file_path)




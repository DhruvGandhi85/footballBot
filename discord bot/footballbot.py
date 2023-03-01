import discord
import requests
from bs4 import BeautifulSoup
from prettytable import PrettyTable

intents = discord.Intents.all()
client = discord.Client(command_prefix='!', intents=intents)


@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$player'):
        args = message.content.split('$player ')[1].split()
        first_name = args[0]
        last_name = args[1]
        weeks = int(args[2])
        player = f"{first_name}-{last_name}".lower()
        url = f'https://www.nfl.com/players/{player}/stats/logs'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        stats_tables = soup.find_all('table')
        headers = [th.text.strip() for th in stats_tables[1].select('thead th')]
        orig_headers = headers[1:]
        data_rows = [[td.text.strip() for td in tr.select('td')] for tr in
                     stats_tables[1].select('tbody tr')][:weeks]

        career_stats = {}
        count_cols = {}
        for row in data_rows:
            week = row[0]
            stats = row[1:]
            for i, header in enumerate(orig_headers):
                col_count = count_cols.get(header, 0)
                count_cols[header] = col_count + 1
                if col_count > 0:
                    header = f"{header}_{col_count}"
                career_stats[week] = career_stats.get(week, {})
                career_stats[week][header] = stats[i]

        table = PrettyTable()
        table.field_names = ['Week'] + [header for header in headers[1:]]
        for week, stats in career_stats.items():
            row = [week] + [stats.get(header, '') for header in headers[1:]]
            table.add_row(row)

        await message.channel.send(f'```{table}```')

with open("bot auth token.txt", "r") as file:
    # read the contents of the file
    file_contents = file.read()


client.run(file_contents)

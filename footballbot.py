import os
from typing import List
from contextlib import suppress

import discord
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from bs4 import BeautifulSoup
from scipy.stats import linregress


def get_player_url(first_name: str, last_name: str) -> str:
    player = f"{first_name}-{last_name}".lower()
    return f'https://www.nfl.com/players/{player}/stats/logs'


def get_player_position(headers: List[str]) -> str:
    for i, header in enumerate(headers):
        if i == 4:
            if header == 'REC':
                return 'WR'
            elif header == 'COMP':
                return 'QB'
            elif header == 'ATT':
                return 'RB'
    return ''


def get_renamed_headers(headers: List[str], player_pos: str) -> List[str]:
    for i, header in enumerate(headers):
        if player_pos == 'QB':
            if i in {13, 14, 15, 16}:
                headers[i] = f"RUS_{header}"
            if i in {4, 5, 6, 7, 8}:
                headers[i] = f"PASS_{header}"
        elif player_pos == 'WR':
            if i in {9, 10, 11, 12, 13}:
                headers[i] = f"RUS_{header}"
            if i in {5, 6, 7, 8}:
                headers[i] = f"REC_{header}"
        elif player_pos == 'RB':
            if i in {10, 11, 12}:
                headers[i] = f"REC_{header}"
            if i in {4, 5, 6, 7, 8}:
                headers[i] = f"RUS_{header}"
    return headers[1:]


def get_player_stats(first_name: str, last_name: str, weeks: int) -> pd.DataFrame:
    url = get_player_url(first_name, last_name)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    stats_tables = soup.find_all('table')
    try:
        headers = [th.text.strip()
                   for th in stats_tables[1].select('thead th')]
    except Exception as e:
        raise ValueError(
            f"Failed to retrieve stats for {first_name} {last_name}")
    player_pos = get_player_position(headers)
    renamed_headers = get_renamed_headers(headers, player_pos)

    data_rows = [[td.text.strip() for td in tr.select('td')]
                 for tr in stats_tables[1].select('tbody tr')][:weeks]

    career_stats = {}
    for row in data_rows:
        week = row[0]
        stats = row[1:]
        for i, header in enumerate(renamed_headers):
            career_stats[week] = career_stats.get(week, {})
            career_stats[week][header] = stats[i]

    return pd.DataFrame.from_dict(career_stats, orient='index')


def plot_weekly_yards(df_player: pd.DataFrame) -> str:
    yards_cols = [col for col in df_player.columns if 'YDS' in col]

    total_yards = df_player[yards_cols].apply(
        pd.to_numeric, errors='coerce').sum(axis=1).astype(int)

    # Filter out rows with NaN values
    total_yards = total_yards[~np.isnan(total_yards)]

    # Calculate line of best fit
    x = np.array(total_yards.index).astype(int)
    y = np.array(total_yards).astype(int)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'o')
    plt.plot(x, line, '-')
    plt.title('Weekly Yards')
    plt.xlabel('Week')
    plt.ylabel('Yards')
    plt.tight_layout()

    chart_path = 'chart.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path


def calculate_fantasy_points(df_player: pd.DataFrame) -> float:

    # Calculate fantasy points for each week based on the multiplier values
    fantasy_points = []
    multipliers = {'PASS_YDS': 0.04, 'PASS_TD': 4,
                   'INT': -2, 'RUSH_YDS': 0.1, 'RUSH_TD': 6, 'REC_YDS': 0.1, 'REC_TD': 6, 'FUM': -1, 'LOST': -1, 'REC': 1}
    for index, row in df_player.iterrows():
        fp = 0
        for col, value in row.items():
            if col in multipliers and value != '':
                fp += float(value) * multipliers[col]  # convert value to float
        fantasy_points.append(fp)

    return fantasy_points


def plot_fantasy_points_over_time(df_player: pd.DataFrame, fantasy_points: List[float]) -> str:

    # Plot the fantasy points over time
    x = np.array(df_player.index).astype(int)
    y = np.array(fantasy_points).astype(float)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'o')
    plt.plot(x, line, '-')
    plt.title('Fantasy Points Over Time')
    plt.xlabel('Week')
    plt.ylabel('Fantasy Points')
    plt.tight_layout()

    chart_path = 'fantasy_points_chart.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path


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
        if len(args) == 3:
            first_name, last_name, weeks = args[0], args[1], int(args[2])
        elif len(args) == 2:
            first_name, last_name, weeks = args[0], args[1], 5
        try: 
            df_player = get_player_stats(first_name, last_name, weeks)
        except ValueError as e:
            await message.channel.send(f"Failed to retrieve stats for {first_name} {last_name}")            
            return
        player_stats = calculate_fantasy_points(df_player)

        # Send the player stats as a formatted message
        await send_player_stats(message.channel, first_name, last_name, weeks, df_player)

        # Send the yards chart as an attachment
        yds_chart_path = plot_weekly_yards(df_player)
        await send_chart_as_attachment(message.channel, yds_chart_path)

        # Send the fantasy points chart as an attachment
        pts_chart_path = plot_fantasy_points_over_time(df_player, player_stats)
        await send_chart_as_attachment(message.channel, pts_chart_path)

        # Remove the chart files
        os.remove(yds_chart_path)
        os.remove(pts_chart_path)


async def send_player_stats(channel, first_name, last_name, weeks, df_player):

    output = f"**{first_name} {last_name} - Stats for last {weeks} weeks:**\n\n"
    for index, row in df_player.iterrows():
        output += f"Week {index}:\n"
        for col, value in row.items():
            output += f"{col}: {value} | "
        output += "\n\n"
    messages = [output[i:i+2000] for i in range(0, len(output), 2000)]
    for i, msg in enumerate(messages):
        if i == 0:
            await channel.send(msg)
        else:
            await channel.send(msg[:1997] + '...')


async def send_chart_as_attachment(channel, chart_path):

    with open(chart_path, 'rb') as f:
        chart = discord.File(f)
        await channel.send(file=chart)


with open("bot auth token.txt", "r") as file:
    file_contents = file.read()
client.run(file_contents)

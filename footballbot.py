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
from discord.ext import commands


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

    non_zero_indices = np.nonzero(y)  # get the indices where y is non-zero
    non_zero_x = x[non_zero_indices]
    non_zero_y = y[non_zero_indices]

    slope, intercept, r_value, p_value, std_err = linregress(
        non_zero_x, non_zero_y)
    line = slope * x + intercept

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'o', color='blue')
    plt.plot(x, line, '-', color='green', linewidth=2)
    plt.title('Weekly Yards', fontsize=18)
    plt.xticks(np.arange(x.min(), x.max()+1))
    plt.xlabel('Week', fontsize=14)
    plt.ylabel('Yards', fontsize=14)
    plt.tight_layout()

    chart_path = 'chart.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path


def calculate_fantasy_points(df_player: pd.DataFrame) -> List[float]:

    # Calculate fantasy points for each week based on the multiplier values
    fantasy_points = []
    multipliers = {'PASS_YDS': 0.04, 'PASS_TD': 4,
                   'INT': -2, 'RUS_YDS': 0.1, 'RUS_TD': 6, 'REC': 1, 'REC_YDS': 0.1, 'REC_TD': 6, 'FUM': -1, 'LOST': -1}
    for index, row in df_player.iterrows():
        fp = 0
        for col, value in row.items():
            if col in multipliers and value != '':
                fp += float(value) * multipliers[col]  # convert value to float
        fantasy_points.append(fp)
    return fantasy_points


def plot_fantasy_points(df_player: pd.DataFrame, fantasy_points: List[float]) -> str:

    # Plot the fantasy points over time
    x = np.array(df_player.index).astype(int)
    y = np.array(fantasy_points).astype(float)

    non_zero_indices = np.nonzero(y)  # get the indices where y is non-zero
    non_zero_x = x[non_zero_indices]
    non_zero_y = y[non_zero_indices]

    slope, intercept, r_value, p_value, std_err = linregress(
        non_zero_x, non_zero_y)
    line = slope * x + intercept

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', color='blue', markersize=8, label='Fantasy Points')
    plt.plot(x, line, '-', color='orange', linewidth=2, label='Trend Line')
    plt.title('Fantasy Points by Week (PPR)', fontsize=18)
    plt.xlabel('Week', fontsize=14)
    plt.ylabel('Fantasy Points', fontsize=14)
    plt.xticks(np.arange(x.min(), x.max()+1))
    plt.tight_layout()

    chart_path = 'fantasy_points_chart.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
bot = commands.Bot(command_prefix='.', intents=intents)


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
        player_points = calculate_fantasy_points(df_player)

        # Create the view with three buttons
        view = discord.ui.View()
        statsButton = discord.ui.Button(label="Player Stats")
        ydsButton = discord.ui.Button(label="Weekly Yards Plot")
        ptsButton = discord.ui.Button(label="Fantasy Points Plot")

        # Define the callback functions for each button

        async def statsCallback(interaction: discord.Interaction):
            await send_player_stats(message.channel, first_name, last_name, weeks, df_player)

        async def ydsCallback(interaction: discord.Interaction):
            yds_chart_path = plot_weekly_yards(df_player)
            await send_chart_as_attachment(message.channel, yds_chart_path)
            os.remove(yds_chart_path)

        async def ptsCallback(interaction: discord.Interaction):
            pts_chart_path = plot_fantasy_points(df_player, player_points)
            await send_chart_as_attachment(message.channel, pts_chart_path)
            os.remove(pts_chart_path)

        # Assign the callback functions to the buttons
        statsButton.callback = statsCallback
        ydsButton.callback = ydsCallback
        ptsButton.callback = ptsCallback

        # Add the buttons to the view
        view.add_item(statsButton)
        view.add_item(ydsButton)
        view.add_item(ptsButton)
        view.add_item(discord.ui.Button(label="NFL.com page",
                      style=discord.ButtonStyle.link, url=get_player_url(first_name, last_name)))

        # Send the message with the view
        output = f"**{first_name.capitalize()} {last_name.capitalize()} - Overview for last {weeks} weeks**\n\n"
        yards_cols = [col for col in df_player.columns if 'YDS' in col]
        total_yards = df_player[yards_cols].apply(
            pd.to_numeric, errors='coerce').sum(axis=1).astype(int)

        # Filter out rows with NaN values
        total_yards = total_yards[~np.isnan(total_yards)]
        output += f"Total Yards: {total_yards.sum()}\n"
        output += f"Avg Yards (disregarding DNP): {np.ma.masked_equal(total_yards, 0).mean():.2f}\n"
        output += f"Total Fantasy Points: {sum(player_points):.2f}\n"
        output += f"Avg Fantasy Points (disregarding DNP): {sum(player_points)/len([p for p in player_points if p != 0]):.2f}\n"
        output += f"Please select an option:"
        await message.channel.send(content=output, view=view)


async def send_player_stats(channel, first_name, last_name, weeks, df_player):
    output = f"**{first_name.capitalize()} {last_name.capitalize()} - Stats for last {weeks} weeks:**\n\n"
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

with open(".env", "r") as token:
    token = token.read()
client.run(token)

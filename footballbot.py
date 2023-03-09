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


def get_player_url(first_name: str, last_name: str, year: int) -> str:
    player = f"{first_name}-{last_name}".lower()
    return f'https://www.nfl.com/players/{player}/stats/logs/{year}/'


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


def get_player_stats(first_name: str, last_name: str, weeks: int, start_year: int, end_year: int) -> pd.DataFrame:
    career_stats = {}
    for year in range(start_year, end_year+1):
        url = get_player_url(first_name, last_name, year)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        stats_tables = soup.find_all('table')
        try:
            if year == 2020:
                headers = [th.text.strip()
                           for th in stats_tables[0].select('thead th')]
            else:
                headers = [th.text.strip()
                           for th in stats_tables[1].select('thead th')]
        except Exception as e:
            raise ValueError(
                f"Failed to retrieve stats for {first_name} {last_name}")
        player_pos = get_player_position(headers)
        renamed_headers = get_renamed_headers(headers, player_pos)

        if year == 2020:
            data_rows = [[td.text.strip() for td in tr.select('td')]
                         for tr in stats_tables[0].select('tbody tr')][:weeks]
        else:
            data_rows = [[td.text.strip() for td in tr.select('td')]
                         for tr in stats_tables[1].select('tbody tr')][:weeks]

        for row in data_rows:
            week = f"{row[0]}-{year}"
            stats = row[1:]
            for i, header in enumerate(renamed_headers):
                career_stats[week] = career_stats.get(week, {})
                career_stats[week][header] = stats[i]

    career_df = pd.DataFrame.from_dict(career_stats, orient='index')

    return career_df


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


def plot_weekly_yards(df_player: pd.DataFrame) -> List[str]:
    df = df_player.copy()
    yards_cols = [col for col in df_player.columns if 'YDS' in col]
    total_yards = df_player[yards_cols].apply(
        pd.to_numeric, errors='coerce').sum(axis=1).astype(int)

    # Filter out rows with NaN values
    total_yards = total_yards[~np.isnan(total_yards)]
    df['Total Yards'] = total_yards
    df['Week'], df['Year'] = df.index.str.split('-').str
    grouped = df.groupby('Year')

    chart_paths = []
    for year, year_df in grouped:
        x_weeks = year_df['Week'].astype(int)
        y = year_df['Total Yards']

        slope, intercept, r_value, p_value, std_err = linregress(
            x_weeks, y)
        line = slope * x_weeks + intercept

        plt.figure(figsize=(10, 6))
        plt.plot(x_weeks, y, 'o', color='blue')
        plt.plot(x_weeks, line, '-', color='green', linewidth=2)
        plt.title(f'Total Yards per Week - {year}', fontsize=18)
        plt.xlabel('Week', fontsize=14)
        plt.ylabel('Total Yards', fontsize=14)
        # Set tick locations and labels
        x_ticks = np.unique(x_weeks)
        plt.xticks(x_ticks)
        plt.tight_layout()

        chart_path = f'fantasy_points_chart_{year}.png'
        plt.savefig(chart_path)
        plt.close()
        chart_paths.append(chart_path)

    return chart_paths


def plot_fantasy_points(df_player: pd.DataFrame, fantasy_points: List[float]) -> List[str]:
    # Group the data by week and year, and take the mean of the fantasy points for each group
    df = df_player.copy()
    df['Fantasy Points'] = fantasy_points
    df['Week'], df['Year'] = df.index.str.split('-').str
    df['Fantasy Points'] = pd.to_numeric(df['Fantasy Points'])
    grouped = df.groupby('Year')

    chart_paths = []
    for year, year_df in grouped:
        x_weeks = year_df['Week'].astype(int)
        y = year_df['Fantasy Points']

        slope, intercept, r_value, p_value, std_err = linregress(
            x_weeks, y)
        line = slope * x_weeks + intercept

        plt.figure(figsize=(10, 6))
        plt.plot(x_weeks, y, 'o', color='blue',
                 markersize=8, label='Fantasy Points')
        plt.plot(x_weeks, line, '-', color='orange',
                 linewidth=2, label='Trend Line')
        plt.title(f'Fantasy Points by Week (PPR) - {year}', fontsize=18)
        plt.xlabel('Week', fontsize=14)
        plt.ylabel('Fantasy Points', fontsize=14)

        # Set tick locations and labels
        x_ticks = np.unique(x_weeks)
        plt.xticks(x_ticks)

        plt.tight_layout()
        chart_path = f'fantasy_points_chart_{year}.png'
        plt.savefig(chart_path)
        plt.close()
        chart_paths.append(chart_path)

    return chart_paths


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
        if len(args) < 2 or len(args) > 5:
            await message.channel.send(f"Invalid Input. Please use the following format: $player <first_name> <last_name> [<weeks> [<start_year> <end_year>]]")
            return
        first_name, last_name = args[0], args[1]
        weeks = 18
        start_year = end_year = 2022
        if len(args) >= 3:
            if int(args[2]) > 18 or int(args[2]) < 1:
                if int(args[2]) > 1900 and int(args[2]) < 2100:
                    start_year = int(args[2])
                else:
                    await message.channel.send(f"Incorrect number of weeks. Please enter a number between 1 and 18.")
            else:
                weeks = int(args[2])
        if len(args) >= 4:
            if int(args[3]) > 1900 and int(args[3]) < 2100:
                end_year = int(args[3])
            else:
                start_year = int(args[3])
        if len(args) == 5:
            end_year = int(args[4])
        try:
            df_player = get_player_stats(
                first_name, last_name, weeks, start_year, end_year)
        except ValueError as e:
            await message.channel.send(f"Failed to retrieve stats for {first_name} {last_name} in {start_year}-{end_year}. Make sure the player name is spelled correctly and the years are valid. You do not need to account for number of games a season.")
            return
        player_points = calculate_fantasy_points(df_player)

        # Create the view with three buttons
        view = discord.ui.View()
        statsButton = discord.ui.Button(label="Player Stats")
        ydsButton = discord.ui.Button(label="Weekly Yards Plot")
        ptsButton = discord.ui.Button(label="Fantasy Points Plot")

        # Define the callback functions for each button

        async def statsCallback(interaction: discord.Interaction):
            await send_player_stats(message.channel, df_player)

        async def ydsCallback(interaction: discord.Interaction):
            yds_chart_path = plot_weekly_yards(df_player)
            await send_charts_as_attachments(message.channel, yds_chart_path)
            for i in range(len(yds_chart_path)):
                os.remove(yds_chart_path[i])

        async def ptsCallback(interaction: discord.Interaction):
            pts_chart_path = plot_fantasy_points(df_player, player_points)
            await send_charts_as_attachments(message.channel, pts_chart_path)
            for i in range(len(pts_chart_path)):
                os.remove(pts_chart_path[i])

        # Assign the callback functions to the buttons
        statsButton.callback = statsCallback
        ydsButton.callback = ydsCallback
        ptsButton.callback = ptsCallback

        # Add the buttons to the view
        view.add_item(statsButton)
        view.add_item(ydsButton)
        view.add_item(ptsButton)
        view.add_item(discord.ui.Button(label="Player News",
                      style=discord.ButtonStyle.link, url=f'https://www.nfl.com/players/{first_name}-{last_name}/'))

        # Send the message with the view
        if start_year == end_year:
            output = f"**{first_name.capitalize()} {last_name.capitalize()} - Overview for last {weeks} weeks per year from {start_year} **\n\n"
        else:
            output = f"**{first_name.capitalize()} {last_name.capitalize()} - Overview for last {weeks} weeks per year from {start_year} to {end_year}**\n\n"
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


async def send_player_stats(channel, df_player):
    output = ""
    for index, row in df_player.iterrows():
        output += f"Week {index}:\n"
        row_output = ""
        for col, value in row.items():
            row_output += f"{col}: {value} | "
        # Check if adding the current row to the current message will exceed the message length limit
        if len(output) + len(row_output) > 2000:
            # If it will, send the current message and start a new one with the current row
            await channel.send("" + output + "")
            output = row_output + "\n\n"
        else:
            # If it won't, append the current row to the current message
            output += row_output + "\n\n"
    await channel.send("" + output + "")


async def send_charts_as_attachments(channel, chart_paths):
    for chart_path in chart_paths:
        with open(chart_path, 'rb') as f:
            chart = discord.File(f)
            await channel.send(file=chart)


token = os.environ['TOKEN']
client.run(token)

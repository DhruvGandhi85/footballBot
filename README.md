# footballBot
A Discord bot that can be used to get a player's stats over a period of time, as well as their fantasy point output. Stats are scraped from [NFL.com](http://www.nfl.com/). 

## Getting Started
Type '$player first_name last_name weeks start_year end_year' to get a player's stats over a period of time. 

Possible combinations of arguments:
- `$player first_name last_name` for all games within the last season.
- `$player first_name last_name weeks` for "weeks" number of games within the last season.
- `$player first_name last_name start_year` for all games from start_year to last season.
- `$player first_name last_name start_year end_year` for all games from start_year to end_year.
- `$player first_name last_name weeks start_year` for "weeks" number of games from start_year to last season.
- `$player first_name last_name weeks start_year end_year` for "weeks" number of games from start_year to end_year.

https://github.com/DhruvGandhi85/footballBot/assets/37405800/93317f0b-cef4-4393-bc3d-c74d74e12f7c



Hosted on Heroku. Raspberry Pi hosting coming soon.

<!-- heroku ps:scale worker=1 -a football-bot-disc -->

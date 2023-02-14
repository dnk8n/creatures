# ChatGPT creates a battle simulation

## My prompt

Using Python code, can you  help me prototype a simulation in a fictional universe, where creatures battle each other?

My idea is to model:
- `class Environment`
- `class Creature`
- `class Strategy{X}(Creature)` (where {X} is a unique title, external users can submit this type of class, it inherits from Creature class)

Here are some characteristics:
 - each creature is an instantiation of the Creature class
 - each creature is instantiated with an HP of 100
 - 20 instances of each creature are instantiated
 - There are a maximum of 100 games. Each game consists of multiple turns. A turn can end with a creature running without tripping, or if a creature reaches 0 hp. All pairs wait for all other pairs to end their turn before new creatures are re-spawned, and then new game begins. HP does not get reset from game to game
 - Each game there is a random shuffling of creatures, who are faced off to each other in pairs
 - upon HP == 0, the creature with such score is discarded, and the creature class with most collective hp points is instantiated (re-spawned).
 - The environment class keeps collective score of sum of all alive creatures hp and ranks them (so as to calculate values for plots and to re-spawn the correct new creature)
 - In case of any ties, select randomly amongst those that tie
 - The environment class also has methods which create relevant real-time bar graph and line chart plots of collective creature hp per creature class per game (scale suitably refreshed)

- At each turn there is a starter and a follower (selected randomly)
- The actions allowed at each turn are attack, or run
- If attack, there is a 40% chance of a hit, and a 60% chance of a miss
- If run, there is a 20% chance of a trip, and a 80% chance of an escape
- If strike and hit, then 1HP is transferred from the victim to the attacker
- If run and trip, then 1HP is transferred from the escaper to the remainer

I additionally need you to help me think of things I may have missed, and let me know which parts those are. You may ask me questions for further detail.

I am going to ask the solution in 6 parts:
 1. import statements
 2. class Environment
 3. class Creature
 4. class StrategyAlwaysRun
 5. class StrategyAlwaysAttack
 6. class StrategyTitForTat

## The reply

You will be able to see the initial reply by looking back at the git history of this repository. I doubt that the initial answer will run fully. In previous trials I recieved responses that missed initialization of new variables, etc. In one case instead of simply providing a fix, it gave me an entire new version which did run and plot graphs, etc but it didn't look like a good model. That experience got me to thinking about an educational simulation environment, where other users can program their creatures and daily automation can keep track of a leaderboard.

Here is a plantUML schematic that it created based on its initial responses. You might notice some flaws in its design. I hope to implement a much more simplified and robust version.

![Battle Simulation](https://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.githubusercontent.com/dnk8n/creatures/main/docs/schematic.uml.txt&fmt=svg)
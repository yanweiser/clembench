# Grids
Resources for matchit_ascii contain (apart from the necessary prompt templates) all possible grids in [grids_matchit.json](grid_pairs/grids_matchit.json) with IDs to identify them. They are based on the grids from reference game (only from the categories *diagonals*, *letters* and *shapes*) and were manually extended by inverting, turning and mirroring the original grids where it made sense. Additionally for each of those grids there is a grid with an edit distance of two (for difficulty category similar_grid_2).

**Note:** ID 52 and 54 are missing, because they were duplicates. 

[grid-pairs.csv](grid_pairs/grid-pairs.csv) is a table pairing up the aforementioned grid ids into the difficulty categories. For each pair there is also a record of the action which would transform one grid to the other. These actions are:
- vflip: mirror along the vertical axis
- hflip: mirror along the horizontal axis
- add: one of the grids can be achieved by adding more Xs to the other grid
- inv: inversion (Xs become squares and vice versa)
- turn: turning a grid by 90 degrees
- edit2: invert two random positions in the grid

For the used difficulties, the following actions were chosen:
- same_grid: none
- similar_grid_1: vflip, hflip, turn
- similar_grid_2: edit_2
- different_grid: none

async function loadGames(){

    const res = await fetch("http://127.0.0.1:8000/predictions");

    const games = await res.json();

    const container = document.getElementById("games");

    container.innerHTML = "";

    games.forEach(game => {

        const homeProb = game.home_win_probability * 100;
        const awayProb = 100 - homeProb;

        const card = document.createElement("div");
        card.className = "game-card";

        card.innerHTML = `

        <div class="teams">
        ${game.away_team} @ ${game.home_team}
        </div>

        <div class="prob-bar">
            <div class="home-bar" style="width:${homeProb}%"></div>
        </div>

        <div class="stats">
        ${game.home_team} win probability: ${homeProb.toFixed(1)}%
        </div>

        <div class="stats">
        Predicted total runs: ${game.predicted_total_runs.toFixed(1)}
        </div>

        <div class="stats">
        Run differential: ${game.predicted_run_diff.toFixed(1)}
        </div>

        <div class="edge">
        Model Edge: ${(homeProb - 50).toFixed(1)}%
        </div>

        `;

        container.appendChild(card);

    });

}

loadGames();
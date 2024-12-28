// Variables globales pour suivre le jeu
var ai_player_coin = 0; // Pion de l'IA
var current_player_coin = 1; // Pion du joueur actuel
var legal_moves_color = "rgb(127, 255, 0)"; // Couleur des mouvements autorisés

$(document).ready(function(){
    // Cette partie génère la grille complète du jeu Othello
    // Récupère la hauteur de la div "content" et définit la grille à la même taille
    $('#othello_grid').height($('#content').height());
    $('#othello_grid').width($('#content').height());
    // Prépare la grille avec les pions
    var x = "";
    for(var i = 63; i >= 0; i--){
        x = x + "<div class='othello_grid_item' id='othello_grid_" + 
            i + "' onclick='on_othello_grid_click(" + i + 
            ")'><span class='othello_coin' id='othello_coin_" + i + "'></span></div>";  
    }
    // Ajoute la grille dans le HTML
    $('#othello_grid').html(x);

    // Définit la valeur de difficulté
    diff_val();
});

function diff_val(){
    // Met à jour l'affichage du niveau de difficulté de l'IA
    var x = $("#ai_slider").val();
    $("#ai_slider_val").html("Sélectionnez le niveau de difficulté de l'IA (Actuel " + x + "/10)");
}

async function ai_choice(c){
    /* c correspond à l'adversaire IA choisi, voir game_server.py pour les définitions correctes.
    Fonction asynchrone car on attend des données d'un appel AJAX */
    // Envoie le choix d'IA au serveur
    data = await $.ajax({
                data : {ai_player : c, difficulty: $("#ai_slider").val()},
                type : 'POST',
                url : '/ai_choice'
            });
    // Met à jour le titre avec le nom de l'IA
    $("#page_title").html("Jeu Othello contre " + data['ai_player_name']);
    // Affiche les options de sélection des pions
    $("#right_panel").html(data['new_html']);
}

async function coin_choice(c){
    /* c représente le choix de pion : blanc, noir ou aléatoire.
    Fonction asynchrone car on attend des données d'un appel AJAX */
    // Envoie le choix de couleur au serveur
    data = await $.ajax({
        data : {color : c},
        type : 'POST',
        url : '/coin_choice'
    });
    // Définit le pion de l'IA
    ai_player_coin = data['ai_player_coin'];
    // Obtient le joueur qui doit jouer
    current_player_coin = data['player'];
    // Met à jour la grille
    refresh_board(data['white_board'], data['black_board'], 
                  data['legal_moves'], data['done'], data['player']);
    // Affiche le score et retire les options de choix des pions
    $("#right_panel").html(data['score_display_html']);
    // Indique de qui c'est le tour, l'IA joue si c'est à son tour
    if(current_player_coin == ai_player_coin){
        $("#score_display_turn").html("Tour de l'IA");
        ai_step();
    }
}

async function score_set(white_board, black_board){
    /* Met à jour les scores après le rafraîchissement de la grille
    selon la longueur des listes */
    if(ai_player_coin == 0){
        $("#score_ai").html(black_board.length);
        $("#score_you").html(white_board.length);
    } else {
        $("#score_ai").html(white_board.length);
        $("#score_you").html(black_board.length);        
    }
}

async function ai_step(){
    /* Étape de l'IA : fonction séparée pour exécuter plusieurs coups successifs.
    Pause entre chaque coup pour une meilleure expérience utilisateur */
    await sleep(500);
    // Envoie une requête pour demander le coup de l'IA
    data = await $.ajax({
                data : {position : -1},
                type : 'POST',
                url : '/step'
            });
    // Met à jour le joueur actuel
    current_player_coin = data['player'];
    // Rafraîchit la grille
    refresh_board(data['white_board'], data['black_board'], 
                  data['legal_moves'], data['done'], data['player']);
    // Si c'est encore le tour de l'IA, rejoue
    if(data['player'] == ai_player_coin){
        ai_step();
    } 
}

async function refresh_board(white_board, black_board, legal_moves, 
                             done, player){
    /* Met à jour les couleurs des pions selon les données reçues du serveur

    white_board : indices des pions blancs
    black_board : indices des pions noirs
    legal_moves : indices des mouvements autorisés
    done : 0/1, indique si le jeu est terminé
    player : 0/1, joueur qui doit jouer ensuite */
    // Réinitialise toutes les couleurs de la grille
    await $(".othello_coin").css({"background-color": "transparent"});
    // Met à jour les couleurs selon les données
    if(white_board.length > 0){
        await set_color("#F2F2F2", white_board);
    }
    if(black_board.length > 0){
        await set_color("#000000", black_board);
    }
    if(legal_moves.length > 0){
        await set_color(legal_moves_color, legal_moves);
    }
    // Met à jour les scores
    await score_set(white_board, black_board);
    // Affiche de qui c'est le tour
    if(ai_player_coin == player){
        $("#score_display_turn").html("Tour de l'IA");
    } else {
        $("#score_display_turn").html("Votre tour");
    }
    // Affiche le gagnant si le jeu est terminé
    if(done == 1){
        $("#score_display_turn").html("Partie terminée");
        $("#winner").css('display', 'block');
        if(white_board.length == black_board.length){
            $("#winner").html("Égalité");
        } else if(white_board.length > black_board.length){
            $("#winner").html(ai_player_coin == 1 ? "Gagnant : IA" : "Gagnant : Vous");
        } else {
            $("#winner").html(ai_player_coin == 0 ? "Gagnant : IA" : "Gagnant : Vous");
        }
    }
}

function set_color(c, l){
    /* Définit la couleur c pour une liste d'indices l */
    for (var i = 0; i < l.length; i++) {
        $("#othello_coin_" + l[i]).css("background-color", c);
        if(c == legal_moves_color){
            $("#othello_coin_" + l[i]).css("opacity", 0.6);
        } else {
            $("#othello_coin_" + l[i]).css("opacity", 1);
        }
    }
}

async function on_othello_grid_click(i){
    /* Gestionnaire d'événements déclenché lors du clic sur une cellule de la grille */
    if($("#othello_coin_" + i).css("background-color") == legal_moves_color &&
       ai_player_coin != current_player_coin){
        // Envoie le coup joué par le joueur au serveur
        data = await $.ajax({
                    data : {position : i},
                    type : 'POST',
                    url : '/step'
            });
        // Met à jour le joueur actuel
        current_player_coin = data['player'];
        // Rafraîchit la grille
        refresh_board(data['white_board'], data['black_board'], 
                      data['legal_moves'], data['done'], data['player']);
        // Si c'est le tour de l'IA, elle joue
        if(current_player_coin == ai_player_coin){
            ai_step();
        }
    }
}

function sleep(ms) {
    /* Fonction pour attendre avant d'exécuter une action */
    return new Promise(resolve => setTimeout(resolve, ms));
}

function fetchMoves(filename) {
    fetch(`/get_moves/${filename}`)
        .then(response => response.text())
        .then(data => {
            document.getElementById('moves_display').innerText = data;
        })
        .catch(error => console.error('Erreur:', error));
}

function fetchScores(filename) {
    fetch(`/get_scores/${filename}`)
        .then(response => response.text())
        .then(data => {
            const scores = data.split('\n');
            document.getElementById('score_ai').innerText = scores[0].split(': ')[1];
            document.getElementById('score_you').innerText = scores[1].split(': ')[1];
        })
        .catch(error => console.error('Erreur:', error));
}

// Exemple d'utilisation
fetchMoves('moves_0001.txt');
fetchScores('score_0001.txt');
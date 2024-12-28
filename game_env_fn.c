// run this command to build the shared library
// gcc -fPIC -shared -o game_env_fn.dll game_env_fn.c
// #include <stdio.h>

unsigned int get_total_set_bits(unsigned long long s){
    /* La méthode de Brian Kernighan passe par autant d'itérations qu'il y a de bits actifs.
    Ainsi, si nous avons un mot de 32 bits avec seulement le bit haut activé, la boucle passera une seule fois. */
    unsigned int c; // c accumule le total des bits actifs dans s
    for (c = 0; s; c++){
        s &= s - 1; // efface le bit le moins significatif activé
    }
    return c;
}

void get_set_bits_array(unsigned long long s, 
                        int arr[]){
    /* Modifie le tableau pour stocker les bits actifs individuels
    arr -> tableau à modifier, il doit être initialisé à la longueur requise au préalable (cela ne peut pas être dynamique) */
    unsigned int count = 0, idx = 0;
    while(s){
        if(s&1){
            arr[idx] = count; idx++;
        }
        s = s >> 1; count++;
    }
}

void get_next_board(unsigned long long s0, unsigned long long s1, 
                    unsigned int p, unsigned long long a,
                    unsigned long long* ns0, unsigned long long* ns1){
    /* Détermine le bitboard mis à jour après avoir effectué l'action a
    en utilisant le joueur passé à la fonction

    Paramètres
    ----------
    s : liste
        contient les bitboards pour les pièces blanches, noires et 
        le joueur actuel
    a : int (64 bits)
        le bit où l'action doit être effectuée est défini sur 1

    Retourne
    -------
    s_next : liste
        bitboards mis à jour
    */
    unsigned long long board_p = s0, board_notp = s1;
    if(p){
        board_p = s1; board_notp = s0;
    }
    // garde un maître global des mises à jour
    unsigned long long update_master = 0, m, c;
    // gauche
    m = 0;
    c = board_notp & (a << 1) & 18374403900871474942llu;
    while(c & board_notp){
        m = m | c;
        c = (c << 1) & 18374403900871474942llu;        
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // droite
    m = 0;
    c = board_notp & (a >> 1) & 9187201950435737471llu;
    while(c & board_notp){
        m = m | c;
        c = (c >> 1) & 9187201950435737471llu;        
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // haut
    m = 0;
    c = board_notp & (a << 8) & 18446744073709551360llu;
    while(c & board_notp){
        m = m | c;
        c = (c << 8) & 18446744073709551360llu;        
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // bas
    m = 0;
    c = board_notp & (a >> 8) & 72057594037927935llu;
    while(c & board_notp){
        m = m | c;
        c = (c >> 8) & 72057594037927935llu;        
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // droite_haut
    m = 0;
    c = board_notp & (a << 7) & 9187201950435737344llu;
    while(c & board_notp){
        m = m | c;
        c = (c << 7) & 9187201950435737344llu;
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // gauche_haut
    m = 0;
    c = board_notp & (a << 9) & 18374403900871474688llu;
    while(c & board_notp){
        m = m | c;
        c = (c << 9) & 18374403900871474688llu;
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // droite_bas
    m = 0;
    c = board_notp & (a >> 9) & 35887507618889599llu;
    while(c & board_notp){
        m = m | c;
        c = (c >> 9) & 35887507618889599llu;
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // gauche_bas
    m = 0;
    c = board_notp & (a >> 7) & 71775015237779198llu;
    while(c & board_notp){
        m = m | c;
        c = (c >> 7) & 71775015237779198llu;
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    
    // toutes les directions ont été recherchées, maintenant mettre à jour les bitboards
    board_p = board_p | update_master | a;
    board_notp = board_notp - update_master;
    // retour
    if(!p){
        *ns0 = board_p; *ns1 = board_notp;
    }
    else{
        *ns0 = board_notp; *ns1 = board_p;
    }
}


void legal_moves_helper(unsigned long long s0, unsigned long long s1,
                         unsigned int p, unsigned long long* moves){
    /* Obtient le bitboard des mouvements légaux pour le joueur donné

    Paramètres
    ----------
    s : liste
        contient le bitboard pour les pièces noires, blanches et
        l'int représentant le joueur à jouer

    Retourne
    -------
    m : int (64 bits)
        bitboard représentant les mouvements légaux pour le joueur p
    */
    unsigned long long board_p = s0, board_notp = s1;
    if(p){
        board_p = s1; board_notp = s0;
    }
    // garde un maître global des mises à jour
    unsigned long long c, e, m = 0;
    // définit l'ensemble des cases vides
    e = ~(board_p | board_notp);
    // pour chaque direction, exécuter la boucle while pour obtenir les mouvements légaux
    // la boucle while traverse les chemins des pièces de même couleur
    // obtenir l'ensemble des positions où il y a une pièce du joueur opposé
    // dans la direction du joueur à jouer
    // gauche
    c = board_notp & (board_p << 1) & 18374403900871474942llu;
    while(c){
        // si la case immédiatement dans la direction est vide, c'est un mouvement légal
        m = m | (e & (c << 1) & 18374403900871474942llu);
        // on peut continuer la boucle jusqu'à ce qu'on rencontre une pièce du joueur opposé
        c = board_notp & (c << 1) & 18374403900871474942llu;
    }
    // droite
    c = board_notp & (board_p >> 1) & 9187201950435737471llu;
    while(c){
        // si la case immédiatement dans la direction est vide, c'est un mouvement légal
        m = m | (e & (c >> 1) & 9187201950435737471llu);
        // on peut continuer la boucle jusqu'à ce qu'on rencontre une pièce du joueur opposé
        c = board_notp & (c >> 1) & 9187201950435737471llu;
    }
    // haut
    c = board_notp & (board_p << 8) & 18446744073709551360llu;
    while(c){
        // si la case immédiatement dans la direction est vide, c'est un mouvement légal
        m = m | (e & (c << 8) & 18446744073709551360llu);
        // on peut continuer la boucle jusqu'à ce qu'on rencontre une pièce du joueur opposé
        c = board_notp & (c << 8) & 18446744073709551360llu;        
    }
    // bas
    c = board_notp & (board_p >> 8) & 72057594037927935llu;
    while(c){
        // si la case immédiatement dans la direction est vide, c'est un mouvement légal
        m = m | (e & (c >> 8) & 72057594037927935llu);
        // on peut continuer la boucle jusqu'à ce qu'on rencontre une pièce du joueur opposé
        c = board_notp & (c >> 8) & 72057594037927935llu;
    }
    // droite_haut
    c = board_notp & (board_p << 7) & 9187201950435737344llu;
    while(c){
        // si la case immédiatement dans la direction est vide, c'est un mouvement légal
        m = m | (e & (c << 7) & 9187201950435737344llu);
        // on peut continuer la boucle jusqu'à ce qu'on rencontre une pièce du joueur opposé
        c = board_notp & (c << 7) & 9187201950435737344llu;
    }
    // gauche_haut
    c = board_notp & (board_p << 9) & 18374403900871474688llu;
    while(c){
        // si la case immédiatement dans la direction est vide, c'est un mouvement légal
        m = m | (e & (c << 9) & 18374403900871474688llu);
        // on peut continuer la boucle jusqu'à ce qu'on rencontre une pièce du joueur opposé
        c = board_notp & (c << 9) & 18374403900871474688llu;
    }
    // droite_bas
    c = board_notp & (board_p >> 9) & 35887507618889599llu;
    while(c){
        // si la case immédiatement dans la direction est vide, c'est un mouvement légal
        m = m | (e & (c >> 9) & 35887507618889599llu);
        // on peut continuer la boucle jusqu'à ce qu'on rencontre une pièce du joueur opposé
        c = board_notp & (c >> 9) & 35887507618889599llu;
    }
    // gauche_bas
    c = board_notp & (board_p >> 7) & 71775015237779198llu;
    while(c){
        // si la case immédiatement dans la direction est vide, c'est un mouvement légal
        m = m | (e & (c >> 7) & 71775015237779198llu);
        // on peut continuer la boucle jusqu'à ce qu'on rencontre une pièce du joueur opposé
        c = board_notp & (c >> 7) & 71775015237779198llu;
    }
    // retourner les mouvements légaux finaux
    // retour m;
    *moves = m;
}


void step(unsigned long long s0, unsigned long long s1, unsigned int p,
          unsigned long long a, unsigned long long* ns0,
          unsigned long long* ns1, unsigned long long* legal_moves,
          unsigned int* np, unsigned int* done){
    /* Effectue un mouvement sur le plateau à l'emplacement d'action donné
    et vérifie aussi les cas terminaux, si aucun mouvement n'est possible, etc.

    Paramètres
    ----------
    s : tuple
        état actuel du plateau défini par les bitboards pour les pièces noires, blanches
        et le joueur actuel
    a : int (64 bits)
        le bit déterminant la position où jouer est défini sur 1

    Retourne
    -------
    s_next : liste
        bitboards mis à jour pour les pièces noires, blanches, et le joueur suivant
    legal_moves : bitboard
        mouvements légaux pour le prochain joueur
    next_player : int
        si 1, le joueur suivant joue, sinon 0
    done : int
        1 si le jeu est terminé, sinon 0
    */
    *done = 0;
    // variable pour suivre si le jeu est terminé
    // les récompenses seront déterminées par la classe du jeu
    get_next_board(s0, s1, p, a, ns0, ns1);
    // changer de joueur avant de vérifier les mouvements légaux
    *np = 1;
    if(p){
        *np = 0;
    }
    legal_moves_helper(*ns0, *ns1, *np, legal_moves);
    // vérifier si des mouvements légaux sont disponibles
    if(!*legal_moves){
        // soit le joueur actuel ne peut pas jouer, soit le jeu est terminé
        // vérifie effectivement si ns0|ns1 a tous les bits activés
        if(!(~(*ns0 | *ns1))){
            // le jeu est terminé
            *done = 1;
        }   
        else{
            // le joueur actuel ne peut pas jouer, changer de joueur
            *np = 1 - *np;
            // vérifier les mouvements légaux à nouveau
            legal_moves_helper(*ns0, *ns1, *np, legal_moves);
            if(!*legal_moves){
                // aucun mouvement possible, le jeu est terminé
                *done = 1;
            }
            // sinon, le joueur original jouera ensuite et l'autre passera son tour, rien à modifier
        }
    }
    // retour
    // retour s_next, legal_moves, s_next[2], done
}

int get_winner(unsigned long long s0, unsigned long long s1){
    /* Étant donné s0 et s1, les bitboards des pièces noires et blanches, retourne
    0 si les noirs gagnent, 1 si les blancs gagnent, et -1 en cas d'égalité */
    unsigned int b = get_total_set_bits(s0);
    unsigned int w = get_total_set_bits(s1);
    if(b == w){return -1;}
    if(b > w){return 0;}
    return 1;
}

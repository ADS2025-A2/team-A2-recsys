# App/model/recommendations.py

DUMMY_RECOMMENDATIONS = {
    1: [
        {"title": "The Shawshank Redemption", "genre": "Drama"},
        {"title": "The Godfather", "genre": "Crime"},
        {"title": "The Dark Knight", "genre": "Action"},
        {"title": "Pulp Fiction", "genre": "Crime"},
        {"title": "Forrest Gump", "genre": "Drama"},
        {"title": "Inception", "genre": "Sci-Fi"},
        {"title": "Fight Club", "genre": "Drama"},
        {"title": "The Matrix", "genre": "Sci-Fi"},
        {"title": "Gladiator", "genre": "Action"},
        {"title": "Interstellar", "genre": "Sci-Fi"}
    ],
    2: [
        {"title": "Titanic", "genre": "Romance"},
        {"title": "Avatar", "genre": "Sci-Fi"},
        {"title": "Jurassic Park", "genre": "Adventure"},
        {"title": "The Lion King", "genre": "Animation"},
        {"title": "Toy Story", "genre": "Animation"},
        {"title": "The Avengers", "genre": "Action"},
        {"title": "Iron Man", "genre": "Action"},
        {"title": "Black Panther", "genre": "Action"},
        {"title": "Coco", "genre": "Animation"},
        {"title": "Up", "genre": "Animation"}
    ],
    3: [
        {"title": "Joker", "genre": "Drama"},
        {"title": "Deadpool", "genre": "Action"},
        {"title": "Logan", "genre": "Action"},
        {"title": "La La Land", "genre": "Musical"},
        {"title": "Whiplash", "genre": "Drama"},
        {"title": "The Prestige", "genre": "Thriller"},
        {"title": "Dunkirk", "genre": "War"},
        {"title": "Memento", "genre": "Thriller"},
        {"title": "Shutter Island", "genre": "Thriller"},
        {"title": "Inception", "genre": "Sci-Fi"}
    ],
    4: [
        {"title": "Frozen", "genre": "Animation"},
        {"title": "Moana", "genre": "Animation"},
        {"title": "Zootopia", "genre": "Animation"},
        {"title": "Finding Nemo", "genre": "Animation"},
        {"title": "Ratatouille", "genre": "Animation"},
        {"title": "The Incredibles", "genre": "Animation"},
        {"title": "Despicable Me", "genre": "Animation"},
        {"title": "Minions", "genre": "Animation"},
        {"title": "Shrek", "genre": "Animation"},
        {"title": "Kung Fu Panda", "genre": "Animation"}
    ],
    5: [
        {"title": "The Conjuring", "genre": "Horror"},
        {"title": "Insidious", "genre": "Horror"},
        {"title": "It", "genre": "Horror"},
        {"title": "Annabelle", "genre": "Horror"},
        {"title": "The Exorcist", "genre": "Horror"},
        {"title": "Hereditary", "genre": "Horror"},
        {"title": "A Quiet Place", "genre": "Horror"},
        {"title": "Get Out", "genre": "Horror"},
        {"title": "The Shining", "genre": "Horror"},
        {"title": "Paranormal Activity", "genre": "Horror"}
    ],
    6: [
        {"title": "Star Wars: A New Hope", "genre": "Sci-Fi"},
        {"title": "Star Wars: The Empire Strikes Back", "genre": "Sci-Fi"},
        {"title": "Star Wars: Return of the Jedi", "genre": "Sci-Fi"},
        {"title": "Rogue One", "genre": "Sci-Fi"},
        {"title": "The Mandalorian", "genre": "Sci-Fi"},
        {"title": "The Force Awakens", "genre": "Sci-Fi"},
        {"title": "The Last Jedi", "genre": "Sci-Fi"},
        {"title": "The Rise of Skywalker", "genre": "Sci-Fi"},
        {"title": "Obi-Wan Kenobi", "genre": "Sci-Fi"},
        {"title": "Revenge of the Sith", "genre": "Sci-Fi"}
    ],
    7: [
        {"title": "Harry Potter and the Sorcerer's Stone", "genre": "Fantasy"},
        {"title": "Harry Potter and the Chamber of Secrets", "genre": "Fantasy"},
        {"title": "Harry Potter and the Prisoner of Azkaban", "genre": "Fantasy"},
        {"title": "Harry Potter and the Goblet of Fire", "genre": "Fantasy"},
        {"title": "Harry Potter and the Order of the Phoenix", "genre": "Fantasy"},
        {"title": "Harry Potter and the Half-Blood Prince", "genre": "Fantasy"},
        {"title": "Harry Potter and the Deathly Hallows: Part 1", "genre": "Fantasy"},
        {"title": "Harry Potter and the Deathly Hallows: Part 2", "genre": "Fantasy"},
        {"title": "Fantastic Beasts and Where to Find Them", "genre": "Fantasy"},
        {"title": "Fantastic Beasts: The Crimes of Grindelwald", "genre": "Fantasy"}
    ],
    8: [
        {"title": "Finding Dory", "genre": "Animation"},
        {"title": "Kung Fu Panda 2", "genre": "Animation"},
        {"title": "Madagascar", "genre": "Animation"},
        {"title": "Madagascar: Escape 2 Africa", "genre": "Animation"},
        {"title": "Ice Age", "genre": "Animation"},
        {"title": "Ice Age: The Meltdown", "genre": "Animation"},
        {"title": "Ice Age: Dawn of the Dinosaurs", "genre": "Animation"},
        {"title": "Monsters, Inc.", "genre": "Animation"},
        {"title": "Monsters University", "genre": "Animation"},
        {"title": "Big Hero 6", "genre": "Animation"}
    ],
    9: [
        {"title": "The Notebook", "genre": "Romance"},
        {"title": "A Walk to Remember", "genre": "Romance"},
        {"title": "Dear John", "genre": "Romance"},
        {"title": "Safe Haven", "genre": "Romance"},
        {"title": "The Vow", "genre": "Romance"},
        {"title": "Pride & Prejudice", "genre": "Romance"},
        {"title": "Me Before You", "genre": "Romance"},
        {"title": "La La Land", "genre": "Romance"},
        {"title": "The Fault in Our Stars", "genre": "Romance"}
    ],
    10: [
        {"title": "Avengers: Endgame", "genre": "Action"},
        {"title": "Avengers: Infinity War", "genre": "Action"},
        {"title": "Spider-Man: No Way Home", "genre": "Action"},
        {"title": "Doctor Strange", "genre": "Action"},
        {"title": "Black Widow", "genre": "Action"},
        {"title": "Thor: Ragnarok", "genre": "Action"},
        {"title": "Captain Marvel", "genre": "Action"},
        {"title": "Guardians of the Galaxy", "genre": "Action"},
        {"title": "Ant-Man", "genre": "Action"},
        {"title": "Iron Man", "genre": "Action"}
    ]
}

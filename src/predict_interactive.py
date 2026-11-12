"""
Interactive Interface for Match Result Prediction
User can input team stats and get prediction with percentages
"""
import numpy as np
import pickle
import os

# TensorFlow 2.x compatible import
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

# Console colors class (optional, but makes it more visual)
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def load_best_model():
    """Load the best saved model"""
    dropout_values = [0.2, 0.5, 0.8]
    best_model = None
    best_dropout = None

    for dropout in dropout_values:
        model_path = f'../models/best_model_dropout_{dropout}.keras'
        if os.path.exists(model_path):
            if best_model is None:
                best_model = model_path
                best_dropout = dropout

    if best_model:
        print(f"Loading model with Dropout={best_dropout}...")
        return load_model(best_model), best_dropout
    else:
        raise FileNotFoundError("No saved model found")

def load_scaler():
    """Load the scaler for normalization"""
    with open('../data/scaler.pkl', 'rb') as f:
        return pickle.load(f)

def calculate_team_stats():
    """Calculate average team stats from the dataset for reference"""
    import pandas as pd
    df = pd.read_csv('../dataset/Matches.csv')

    stats = {
        'shots_on_target': (df['HomeTarget'].mean(), df['AwayTarget'].mean()),
        'corners': (df['HomeCorners'].mean(), df['AwayCorners'].mean()),
        'fouls': (df['HomeFouls'].mean(), df['AwayFouls'].mean())
    }

    return stats

def get_user_input(team_type, avg_stats):
    """Ask the user for team statistics"""
    print(f"\n{Colors.BOLD}Statistics for {team_type} team:{Colors.END}")
    print(f"(Press Enter to use dataset averages)")

    # Shots on target
    default = avg_stats['shots_on_target'][0 if team_type == 'HOME' else 1]
    inp = input(f"  Shots on target (avg: {default:.1f}): ")
    shots_on_target = float(inp) if inp.strip() else default

    # Corners
    default = avg_stats['corners'][0 if team_type == 'HOME' else 1]
    inp = input(f"  Corners (avg: {default:.1f}): ")
    corners = float(inp) if inp.strip() else default

    # Fouls
    default = avg_stats['fouls'][0 if team_type == 'HOME' else 1]
    inp = input(f"  Fouls (avg: {default:.1f}): ")
    fouls = float(inp) if inp.strip() else default

    return shots_on_target, corners, fouls

def predict_match(model, scaler, home_stats, away_stats):
    """
    Make a match prediction

    Args:
        model: Trained model
        scaler: StandardScaler for normalization
        home_stats: tuple (shots_on_target, corners, fouls)
        away_stats: tuple (shots_on_target, corners, fouls)

    Returns:
        Prediction probabilities
    """
    # Build feature vector (8 features, 4 per team)
    # Order: HomeTarget, HomeCorners, HomeFouls, HomeSaves,
    #        AwayTarget, AwayCorners, AwayFouls, AwaySaves

    home_shots, home_corners, home_fouls = home_stats
    away_shots, away_corners, away_fouls = away_stats

    # Saves = opponent's shots
    home_saves = away_shots
    away_saves = home_shots

    features = np.array([[
        home_shots, home_corners, home_fouls, home_saves,
        away_shots, away_corners, away_fouls, away_saves
    ]])

    # Normalize
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled, verbose=0)[0]

    return prediction

def print_prediction(home_team, away_team, probabilities):
    """Print the prediction in a visual format"""
    prob_home = probabilities[0] * 100
    prob_draw = probabilities[1] * 100
    prob_away = probabilities[2] * 100

    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}PREDICTION: {home_team} vs {away_team}{Colors.END}")
    print("=" * 60)

    # Find the most likely result
    max_prob = max(prob_home, prob_draw, prob_away)

    # Home Win
    color = Colors.GREEN if prob_home == max_prob else Colors.END
    bar = '#' * int(prob_home / 2)
    print(f"{color}{home_team} Win{' ' * (22 - len(home_team))}: {prob_home:5.2f}% {bar}{Colors.END}")

    # Draw
    color = Colors.YELLOW if prob_draw == max_prob else Colors.END
    bar = '#' * int(prob_draw / 2)
    print(f"{color}Draw{' ' * 26}: {prob_draw:5.2f}% {bar}{Colors.END}")

    # Away Win
    color = Colors.BLUE if prob_away == max_prob else Colors.END
    bar = '#' * int(prob_away / 2)
    print(f"{color}{away_team} Win{' ' * (22 - len(away_team))}: {prob_away:5.2f}% {bar}{Colors.END}")

    print("=" * 60)

    # Final prediction
    if prob_home == max_prob:
        print(f"\n{Colors.GREEN}{Colors.BOLD}Prediction: {home_team} WINS{Colors.END}")
    elif prob_draw == max_prob:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Prediction: DRAW{Colors.END}")
    else:
        print(f"\n{Colors.BLUE}{Colors.BOLD}Prediction: {away_team} WINS{Colors.END}")

    print()

def main():
    print("=" * 60)
    print(f"{Colors.BOLD}FOOTBALL RESULT PREDICTOR{Colors.END}")
    print("=" * 60)

    # Load model and scaler
    try:
        model, dropout = load_best_model()
        scaler = load_scaler()
        print(f"{Colors.GREEN}Model loaded successfully{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error loading model: {e}{Colors.END}")
        return

    # Calculate average statistics
    print("\nCalculating average statistics from dataset...")
    avg_stats = calculate_team_stats()

    while True:
        print("\n" + "=" * 60)
        # Ask for team names
        home_team = input(f"\n{Colors.BOLD}HOME team name: {Colors.END}").strip()
        if not home_team:
            print("Exiting...")
            break

        away_team = input(f"{Colors.BOLD}AWAY team name: {Colors.END}").strip()
        if not away_team:
            print("Exiting...")
            break

        # Ask for statistics
        home_stats = get_user_input("HOME", avg_stats)
        away_stats = get_user_input("AWAY", avg_stats)

        # Make prediction
        probabilities = predict_match(model, scaler, home_stats, away_stats)

        # Show result
        print_prediction(home_team, away_team, probabilities)

        # Ask to continue
        continue_pred = input(f"\nMake another prediction? (y/n): ").strip().lower()
        if continue_pred != 'y':
            break

    print("\nThanks for using the predictor!")

if __name__ == "__main__":
    main()

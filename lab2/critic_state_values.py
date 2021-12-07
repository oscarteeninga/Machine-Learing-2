import gym
import tensorflow as tf
import numpy as np


def format_state(state: np.ndarray) -> np.ndarray:
    return np.reshape(state, (1, state.size))


def main() -> None:
    environment = gym.make('CartPole-v1')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model = tf.keras.models.load_model('final.model1')
    model.compile(optimizer=optimizer)
    states = [
        environment.reset()
    ]
    # pozycja wagonika, predkosc wagonika, kat wycyhelnia kijka, predkosc katowa kijka
    for state in states:
        print(state)
        print(float(model(format_state(state))[1][0].numpy()))

    environment.close()


if __name__ == '__main__':
    main()

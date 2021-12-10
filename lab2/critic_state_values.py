import tensorflow_probability as tfp
import gym
import tensorflow as tf
import numpy as np


def format_state(state: list[np.float]) -> np.ndarray:
    return np.reshape(np.array(state), (1, len(state)))


def lunar():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)
    environment = gym.make('LunarLander-v2')
    model = tf.keras.models.load_model('lunar.model')
    model.compile(optimizer=optimizer)
    states = [
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1],
    ]
    for state in states:
        print(format_state(state))
        print(float(model(format_state(state))[1][0].numpy()))
        print(float(tfp.distributions.Categorical(probs=model(format_state(state))[0][0]).experimental_sample_and_log_prob()[0]))

    environment.close()


def cart():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    environment = gym.make('CartPole-v1')
    model = tf.keras.models.load_model('cart.model')
    model.compile(optimizer=optimizer)
    # pozycja wagonika, predkosc wagonika, kat wycyhelnia kijka, predkosc katowa kijka
    states = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    for state in states:
        print(format_state(state))
        print(float(model(format_state(state))[1][0].numpy()))
        print(float(tfp.distributions.Categorical(probs=model(format_state(state))[0][0]).experimental_sample_and_log_prob()[0]))

    environment.close()


def main() -> None:
    # cart()
    lunar()


if __name__ == '__main__':
    main()

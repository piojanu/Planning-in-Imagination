import os
import tempfile
import h5py
import numpy as np

from humblerl import Transition
from humblerl.environments import Discrete, Continuous
from vision import StoreVaeTransitions
from memory import StoreMemTransitions


class TestStoreVaeTransitions(object):
    """Test callback on 3D (e.g. images) and continuous states."""

    def setup(self):
        self.hdf5_file, self.hdf5_path = tempfile.mkstemp()

    def teardown(self):
        os.close(self.hdf5_file)
        os.remove(self.hdf5_path)

    def test_images_states(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((8, 8, 3, 2))
        STATE_SPACE[:] = np.array([0, 255])
        STATE_SPACE_SHAPE = STATE_SPACE.shape[:-1]
        MIN_TRANSITIONS = 96
        CHUNK_SIZE = 48
        N_TRANSITIONS = 1024

        callback = StoreVaeTransitions(STATE_SPACE_SHAPE, self.hdf5_path,
                                       shuffle=False, min_transitions=MIN_TRANSITIONS,
                                       chunk_size=CHUNK_SIZE, dtype=np.uint8)
        transitions = []
        for idx in range(N_TRANSITIONS):
            transition = Transition(
                state=np.random.randint(0, 256, size=(8, 8, 3)),
                action=np.random.choice(ACTION_SPACE),
                reward=np.random.normal(0, 1),
                next_state=np.random.randint(0, 256, size=(8, 8, 3)),
                is_terminal=(idx + 1) % 16 == 0
            )
            transitions.append(transition)
            callback.on_step_taken(idx, transition, None)
        callback.on_loop_end(False)

        h5py_file = h5py.File(self.hdf5_path, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == N_TRANSITIONS // 16
        assert h5py_file.attrs["CHUNK_SIZE"] == CHUNK_SIZE
        assert np.all(h5py_file.attrs["STATE_SHAPE"] == STATE_SPACE_SHAPE)

        for idx, transition in enumerate(transitions):
            assert np.all(h5py_file['states'][idx] == transition.state)

    def test_continous_states(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((4, 2))
        STATE_SPACE[:] = np.array([-1, 1])
        STATE_SPACE_SHAPE = STATE_SPACE.shape[:-1]
        MIN_TRANSITIONS = 96
        CHUNK_SIZE = 48
        N_TRANSITIONS = 1024

        callback = StoreVaeTransitions(STATE_SPACE_SHAPE, self.hdf5_path,
                                       shuffle=False, min_transitions=MIN_TRANSITIONS,
                                       chunk_size=CHUNK_SIZE, dtype=np.float)
        transitions = []
        for idx in range(N_TRANSITIONS):
            transition = Transition(
                state=np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]),
                action=np.random.choice(ACTION_SPACE),
                reward=np.random.normal(0, 1),
                next_state=np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]),
                is_terminal=(idx + 1) % 16 == 0
            )
            transitions.append(transition)
            callback.on_step_taken(idx, transition, None)
        callback.on_loop_end(False)

        h5py_file = h5py.File(self.hdf5_path, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == N_TRANSITIONS
        assert h5py_file.attrs["N_GAMES"] == N_TRANSITIONS // 16
        assert h5py_file.attrs["CHUNK_SIZE"] == CHUNK_SIZE
        assert np.all(h5py_file.attrs["STATE_SHAPE"] == STATE_SPACE_SHAPE)

        for idx, transition in enumerate(transitions):
            assert np.all(h5py_file['states'][idx] == transition.state)

    def test_shuffle_chunks(self):
        ACTION_SPACE = np.array([1, 2, 3])
        STATE_SPACE = np.zeros((4, 2))
        STATE_SPACE[:] = np.array([-1, 1])
        STATE_SPACE_SHAPE = STATE_SPACE.shape[:-1]
        MIN_TRANSITIONS = 48
        CHUNK_SIZE = 48
        N_TRANSITIONS = 48

        callback = StoreVaeTransitions(STATE_SPACE_SHAPE, self.hdf5_path,
                                       shuffle=True, min_transitions=MIN_TRANSITIONS,
                                       chunk_size=CHUNK_SIZE, dtype=np.float)

        states = []
        next_states = []
        transitions = []
        for idx in range(N_TRANSITIONS):
            states.append(np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]).tolist())
            next_states.append(np.random.uniform(STATE_SPACE.T[0], STATE_SPACE.T[1]).tolist())
            transitions.append((np.random.choice(ACTION_SPACE), np.random.normal(0, 1), 0))

            callback.on_step_taken(idx, Transition(
                state=states[-1],
                action=transitions[-1][0],
                reward=transitions[-1][1],
                next_state=next_states[-1],
                is_terminal=transitions[-1][2]
            ), None)

        in_order = True
        h5py_file = h5py.File(self.hdf5_path, "r")
        for idx in range(N_TRANSITIONS):
            state = h5py_file['states'][idx]

            idx_target = states.index(state.tolist())
            if idx != idx_target:
                in_order = False

            assert np.all(h5py_file['states'][idx] == states[idx_target])

        assert not in_order, "Data isn't shuffled!"


class TestStoreMemTransitions(object):

    def setup(self):
        self.hdf5_file, self.hdf5_path = tempfile.mkstemp()

    def teardown(self):
        os.close(self.hdf5_file)
        os.remove(self.hdf5_path)

    def get_random_transition(self, action_space, latent_dim=16, is_terminal=False):
        return Transition(
            state=np.random.uniform(-1, 1, size=(2, latent_dim)),
            action=action_space.sample(),
            reward=np.random.normal(0, 1),
            next_state=np.random.uniform(-1, 1, size=(2, latent_dim)),
            is_terminal=is_terminal
        )

    def test_discrete_action_space(self):
        action_space = Discrete(3)
        latent_dim = 16
        min_transitions = 96
        min_episodes = 96
        chunk_size = 48
        n_transitions = 1024
        n_games = n_transitions // 16

        callback = StoreMemTransitions(self.hdf5_path, latent_dim, action_space,
                                       min_transitions, min_episodes, chunk_size)
        transitions = []
        for idx in range(n_transitions):
            transition = self.get_random_transition(
                action_space, latent_dim, is_terminal=(idx + 1) % 16 == 0)
            transitions.append(transition)
            callback.on_step_taken(idx, transition, None)
        callback.on_loop_end(False)

        h5py_file = h5py.File(self.hdf5_path, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == n_transitions
        assert h5py_file.attrs["N_GAMES"] == n_games
        assert h5py_file.attrs["CHUNK_SIZE"] == chunk_size
        assert h5py_file.attrs["LATENT_DIM"] == latent_dim
        assert h5py_file.attrs["ACTION_DIM"] == 1

        for idx, transition in enumerate(transitions):
            assert np.allclose(h5py_file['states'][idx], transition.state)
            assert h5py_file['actions'][idx][0] == transition.action
            assert np.allclose(h5py_file['rewards'][idx], transition.reward)

        for idx in range(n_games + 1):
            assert h5py_file['episodes'][idx] == idx * 16

    def test_continous_action_space(self):
        action_space = Continuous(num=3, low=np.array([-1.0, 0.0, 0.0]),
                                  high=np.array([1.0, 1.0, 1.0]))
        latent_dim = 16
        min_transitions = 96
        min_episodes = 96
        chunk_size = 48
        n_transitions = 1024
        n_games = n_transitions // 16

        callback = StoreMemTransitions(self.hdf5_path, latent_dim, action_space,
                                       min_transitions, min_episodes, chunk_size)
        transitions = []
        for idx in range(n_transitions):
            transition = self.get_random_transition(
                action_space, latent_dim, is_terminal=(idx + 1) % 16 == 0)
            transitions.append(transition)
            callback.on_step_taken(idx, transition, None)
        callback.on_loop_end(False)

        h5py_file = h5py.File(self.hdf5_path, "r")
        assert h5py_file.attrs["N_TRANSITIONS"] == n_transitions
        assert h5py_file.attrs["N_GAMES"] == n_games
        assert h5py_file.attrs["CHUNK_SIZE"] == chunk_size
        assert h5py_file.attrs["LATENT_DIM"] == latent_dim
        assert h5py_file.attrs["ACTION_DIM"] == action_space.num

        for idx, transition in enumerate(transitions):
            assert np.allclose(h5py_file['states'][idx], transition.state)
            assert np.allclose(h5py_file['actions'][idx], transition.action)
            assert np.allclose(h5py_file['rewards'][idx], transition.reward)

        for idx in range(n_games + 1):
            assert h5py_file['episodes'][idx] == idx * 16

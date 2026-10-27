import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from game import Point, SnakeGameAI


class CollisionRegressionTests(unittest.TestCase):
    def test_tail_cell_is_ignored_when_tail_will_move(self):
        game = SnakeGameAI(w=80, h=100)
        game.snake = [Point(20, 20), Point(20, 40), Point(20, 60)]

        self.assertTrue(game.is_collision(Point(20, 60)))
        self.assertFalse(game.is_collision(Point(20, 60), ignore_tail=True))

    def test_body_cells_are_still_collisions_when_tail_is_ignored(self):
        game = SnakeGameAI(w=80, h=100)
        game.snake = [Point(20, 20), Point(20, 40), Point(20, 0)]

        self.assertTrue(game.is_collision(Point(20, 40), ignore_tail=True))


if __name__ == "__main__":
    unittest.main()

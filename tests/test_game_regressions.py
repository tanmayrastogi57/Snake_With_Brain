import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from game import Direction, Point, SnakeGameAI


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

    def test_food_placement_returns_cleanly_when_board_is_full(self):
        game = SnakeGameAI(w=80, h=20)
        game.snake = [Point(0, 0), Point(20, 0), Point(40, 0), Point(60, 0)]

        self.assertFalse(game._place_food())
        self.assertIsNone(game.food)

    def test_filling_the_board_ends_the_episode(self):
        game = SnakeGameAI(w=80, h=20)
        game.direction = Direction.RIGHT
        game.head = Point(40, 0)
        game.snake = [Point(40, 0), Point(20, 0), Point(0, 0)]
        game.food = Point(60, 0)

        reward, done, score = game.play_step([1, 0, 0])

        self.assertEqual((reward, done, score), (10, True, 1))
        self.assertIsNone(game.food)


if __name__ == "__main__":
    unittest.main()

#include "al2o3_platform/platform.h"
#include "al2o3_os/file.hpp"
#include "accel_sycl.hpp"
#include "worldcylinder.hpp"
#include <curses.h>
#include <stdlib.h>
#include <time.h>

#define DELAYSIZE 200

void myrefresh(void);
void explode(int, int);

short color_table[] =
{
		COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_CYAN,
		COLOR_RED, COLOR_MAGENTA, COLOR_YELLOW, COLOR_WHITE
};

int main() {
	initscr();

	keypad(stdscr, TRUE);
	nodelay(stdscr, TRUE);
	noecho();

	if (has_colors()) start_color();
	for (uint16_t i = 0; i < 8; i++) {
		init_pair(i, color_table[i], COLOR_BLACK);
	}

	using namespace Accel;
	Sycl* sycl = Sycl::Create();

	{
		WorldCylinder world(64, 64);

		while (getch() == ERR)      /* loop until a key is hit */
		{
			attrset(A_NORMAL);
//			erase();

			attrset(COLOR_PAIR(0));
			mvaddstr(0, 0, "-----------------------------");
			mvaddstr(1, 0, "-----------------------------");
			mvaddstr(2, 0, "-----------------------------");
			mvaddstr(3, 0, "-----------------------------");
			mvaddstr(4, 0, "-----------------------------");
			mvaddstr(5, 0, "-----------------------------");
			mvaddstr(6, 0, "-----------------------------");
			mvaddstr(7, 0, "-----------------------------");

			napms(50);
			move(LINES - 1, COLS - 1);
			refresh();
		}
	}

	sycl->Destroy();
	endwin();

	return 0;
}
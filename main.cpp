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
		COLOR_MAGENTA, COLOR_YELLOW, COLOR_BLACK, COLOR_WHITE
};

void display(WorldCylinder const& world) {

	attrset(A_NORMAL);
	for(uint32_t y = 0; y < world.height;++y) {
		for(uint32_t x = 0;x < world.width;++x) {
			float val = world.hostIntensity[(y * world.width) + x];
			char cv = '#';
			if(val > 32 && val < 127) {
				cv = (char) val;
			}
			attrset(COLOR_PAIR((cv-32)/8));
			mvaddch(y,x, cv);
		}


	}
	refresh();
}

int main() {
	WINDOW *pdcWindow = initscr();

	if (pdcWindow != nullptr) {
		keypad(stdscr, TRUE);
		nodelay(stdscr, TRUE);
		noecho();

		if (has_colors())
			start_color();
		for (uint16_t i = 0; i < 8; i++) {
			init_pair(i, color_table[i], COLOR_BLACK);
		}
	}
	using namespace Accel;
	Sycl* sycl = Sycl::Create();

	{
		WorldCylinder world(128, 32);
		world.init(sycl->getQueue());

		int count = 100;
		bool cont = true;
		while (cont)
		{
			if(pdcWindow == nullptr) {
				cont = (count--) ? true : false;
			} else {
				cont = getch() == ERR;
			}

			world.update(sycl->getQueue());

			if(pdcWindow != nullptr) {
				world.flushToHost();
				display(world);
			}
		}
	}

	sycl->Destroy();
	if(pdcWindow != nullptr) {
		endwin();
	}

	return 0;}

compile:
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Server.cpp -o bin/Server.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Model.cpp -o bin/Model.o

	g++ -o bin/server bin/Main.o bin/Model.o bin/Server.o

clear:
	echo "Cleaning bin directory"
	rm -rf bin/*

